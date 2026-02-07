"""
Step06-Spot（単地点予測 UI）
要件：
1) 計算ロジックは 06_predict_apply.py と整合（モデルJSONの features/coef を用いた予測 → B案ロジック）
2) POI は OSM GPKG のみ読み込み、building 属性別に色分け可視化（500m 内集計, Commercial_500 算出）
3) モデル JSON は UI で明示選択（標準/チューニング済み/配列モデルに対応）

出力（UI 内表示のみ／ファイル出力なし）：
- Predicted 人時間/日、台数/日、1台当たり有料時間（期待値）、日収益（有料）
- （参考）RMSE/MAPE が JSON にあれば、人時間の簡易レンジも表示（06_predict_apply と同等）
"""

import os
import json
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Parking Demand Spot Prediction", layout="wide")

# 空間系
try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    gpd = None
    Point = None

# 地図UI（任意）
try:
    from streamlit_folium import st_folium
    import folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

# ====== キャッシュ：データをアップロード状態で用意する ======
@st.cache_resource
def load_preprocessed_data():
    """必要なデータを全て読み込んでメモリに保持"""
    models = {}
    for fname in ["models_bundle_202601151806.json", "models_bundle_202602040438.json"]:
        path = os.path.join("06_result", fname)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                models[fname] = json.load(f)

    osm_gdf = None
    if gpd is not None:
        osm_path = os.path.join("06_result", "Bangkok_building_polygon_mainfaci.gpkg")
        if os.path.exists(osm_path):
            osm_gdf = gpd.read_file(osm_path)
            if osm_gdf.crs is None:
                osm_gdf = osm_gdf.set_crs("EPSG:4326")
            else:
                osm_gdf = osm_gdf.to_crs(epsg=4326)
            if "building" not in osm_gdf.columns:
                osm_gdf["building"] = "unknown"
            else:
                osm_gdf["building"] = osm_gdf["building"].astype(str).str.lower()

    return models, osm_gdf

models, osm_gdf = load_preprocessed_data()

# ====== 06_predict_apply.py と整合する補助関数群 ======
import re

def normalize_models_bundle(bundle: dict):
    """
    Step05 標準（単一）/ tuned（単一）/ models[*]（複数） に両対応
    戻り: (models_list, evaluation, metrics_full)
    """
    if isinstance(bundle.get("models"), list) and bundle["models"]:
        return bundle["models"], bundle.get("evaluation", {}), bundle.get("metrics_full", {})
    m = {
        "label": bundle.get("label", ""),
        "fixed_features": bundle.get("fixed_features", []),
        "formula": {
            "target":  bundle.get("target", ""),
            "features":bundle.get("features", []),
            "coef":    bundle.get("coef", {}),
        }
    }
    return [m], bundle.get("evaluation", {}), bundle.get("metrics_full", {})

def prepare_X_for_formula(df: pd.DataFrame, feats: list) -> pd.DataFrame:
    """06_predict_apply と同等：log_生成、const、欠損0埋め（スポット=単行データ想定）"""
    X = df.copy()
    for fn in feats:
        if isinstance(fn, str) and fn.startswith("log_"):
            base = fn.replace("log_", "")
            v = pd.to_numeric(X.get(base, 0), errors="coerce").fillna(0).clip(lower=1e-6)
            X[fn] = np.log(v)
    if "const" in feats and "const" not in X.columns:
        X["const"] = 1.0
    for c in feats:
        if c not in X.columns:
            X[c] = 0.0
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X[[c for c in feats if c in X.columns]]

def predict_formula(X: pd.DataFrame, formula: dict) -> pd.Series:
    """features×coef の線形和。target が ln(…) なら exp で復元（06_predict_apply と同等）"""
    feats = formula.get("features", [])
    coef  = formula.get("coef", {})
    Xsub = X[[c for c in feats if c in X.columns]].copy()
    for c in feats:
        if c not in Xsub.columns:
            Xsub[c] = 0.0
    Xsub = Xsub[feats]
    beta = pd.Series({k: float(v) for k, v in coef.items()}).reindex(feats, fill_value=0.0)
    ylin = (Xsub * beta.values).sum(axis=1)
    tgt = formula.get("target", "").strip()
    m = re.fullmatch(r"ln\((.+)\)", tgt)
    return np.exp(ylin) if m else ylin

def rmse_mape_from_eval(evaluation: dict, metrics_full: dict) -> tuple[float, float]:
    """CI 参照用（06_predict_apply と同等）：holdout.test → kfold.test → metrics_full の順で取得"""
    rmse = 0.0; mape = 0.0
    try:
        rmse = float(evaluation["holdout"]["test"]["RMSE"])
        mape = float(evaluation["holdout"]["test"]["MAPE"]) / 100.0
        return rmse, mape
    except Exception:
        pass
    try:
        rmse = float(evaluation["kfold"]["test"]["RMSE_mean"])
        mape = float(evaluation["kfold"]["test"]["MAPE_mean"]) / 100.0
        return rmse, mape
    except Exception:
        pass
    try:
        n = max(1, int(metrics_full.get("n", 1)))
        rss = float(metrics_full.get("rss", 0.0))
        rmse = (rss / n) ** 0.5
    except Exception:
        rmse = 0.0
    return rmse, mape

# ====== B案：有料時間/台の推定（06_predict_apply と同等ロジック） ======
def paid_per_car_empirical(p_over, avg_ex):
    p = float(max(0.0, min(1.0, p_over)))
    ex = float(max(0.0, avg_ex))
    return p * ex

def paid_per_car_exponential(avg_stay_hours, free_minutes):
    free_h = float(free_minutes) / 60.0
    mu = float(max(avg_stay_hours, 1e-6))
    return mu * np.exp(-free_h / mu)

def paid_per_car_meandiff(avg_stay_hours, free_minutes):
    free_h = float(free_minutes) / 60.0
    return max(avg_stay_hours - free_h, 0.0)

# ====== OSM: building 別色分け & 500m 内件数 ======
COMMERCIAL_TYPES_7 = {"commercial", "industrial", "kiosk", "office", "retail", "supermarket", "warehouse"}

def read_osm_buildings(gpkg_path: str) -> pd.DataFrame:
    if gpd is None:
        raise RuntimeError("geopandas/shapely が必要です。")
    gdf = gpd.read_file(gpkg_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs(epsg=4326)
    if "building" not in gdf.columns:
        raise ValueError("OSM GPKG に 'building' 列が見つかりません。")
    gdf["building"] = gdf["building"].astype(str).str.lower()
    return gdf

def count_buildings_within_500m(gdf_wgs: "gpd.GeoDataFrame", lat: float, lon: float,
                                building_filter: set | None = None) -> dict:
    """500m 内の building 件数（総数と、Commercial_500）を返す"""
    gdfm = gdf_wgs.to_crs(epsg=3857)
    pt_wgs = gpd.GeoDataFrame(pd.DataFrame({"_": [0]}),
                              geometry=[Point(lon, lat)], crs="EPSG:4326")
    pt_m = pt_wgs.to_crs(epsg=3857).geometry.iloc[0]
    buf = pt_m.buffer(500.0)
    within = gdfm[gdfm.geometry.within(buf)]
    if building_filter:
        mask = within["building"].isin(building_filter)
        commercial_500 = int(mask.sum())
    else:
        commercial_500 = int(within["building"].isin(COMMERCIAL_TYPES_7).sum())
    return {"total_500": int(len(within)), "Commercial_500": commercial_500}

def draw_map_with_buildings(gdf_wgs: "gpd.GeoDataFrame", lat: float, lon: float):
    """building カテゴリ別に色分け可視化。点/中心点で描画。"""
    if not HAS_FOLIUM:
        st.info("`streamlit-folium` が見つからないため簡易表示（st.map）になります。")
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))
        return

    m = folium.Map(location=[lat, lon], zoom_start=15, min_zoom=8)

    # 近傍のみ表示（高速化）
    gdf_plot = gdf_wgs
    try:
        gdfm = gdf_wgs.to_crs(epsg=3857)
        pt_wgs = gpd.GeoDataFrame(pd.DataFrame({"_": [0]}),
                                  geometry=[Point(lon, lat)], crs="EPSG:4326")
        pt_m = pt_wgs.to_crs(epsg=3857).geometry.iloc[0]
        buf = pt_m.buffer(800.0)
        gdf_plot = gdfm[gdfm.geometry.within(buf)].to_crs(epsg=4326)
    except Exception:
        gdf_plot = gdf_wgs

    if len(gdf_plot) == 0:
        st.info("No OSM points found near this location.")
        st_folium(m, height=520, width=900)
        return

    cats = gdf_plot["building"].astype(str).str.lower().unique().tolist()
    palette = [
        "#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00",
        "#a65628","#f781bf","#999999","#66c2a5","#8da0cb",
        "#e78ac3","#a6d854","#ffd92f","#e5c494","#b3b3b3"
    ]
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(sorted(cats))}

    if len(gdf_plot) > 2000:
        gdf_plot = gdf_plot.sample(2000, random_state=42)

    for _, r in gdf_plot.iterrows():
        geom = r.geometry
        if geom is None:
            continue
        p = geom.centroid if geom.geom_type != "Point" else geom
        b = str(r["building"]).lower()
        color = color_map.get(b, "#377eb8")
        folium.CircleMarker(
            location=[p.y, p.x], radius=5, color=color, fill=True, fill_opacity=0.8,
            tooltip=f"building={b}"
        ).add_to(m)

    folium.Marker([lat, lon], icon=folium.Icon(color="red", icon="map-marker")).add_to(m)
    st_folium(m, height=520, width=900)

# ====== Streamlit UI ======
st.title("Step06-Spot: Parking Demand Prediction")

if osm_gdf is None:
    st.error("OSM GPKG not found. Place 06_result/_osm.gpkg.")

# 1) Location input
st.markdown("#### 1) Select a location on the map")
default_lat, default_lon = 13.7563, 100.5018
if "spot_lat" not in st.session_state:
    st.session_state["spot_lat"] = default_lat
if "spot_lon" not in st.session_state:
    st.session_state["spot_lon"] = default_lon

lat = float(st.session_state["spot_lat"])
lon = float(st.session_state["spot_lon"])

if HAS_FOLIUM:
    st.info("Click the map to update latitude/longitude (streamlit-folium).")
    m0 = folium.Map(location=[lat, lon], zoom_start=13)
    folium.Marker([lat, lon], popup="Selected", icon=folium.Icon(color="red")).add_to(m0)
    res_map = st_folium(m0, height=360, width=900)
    if isinstance(res_map, dict) and res_map.get("last_clicked"):
        lat = float(res_map["last_clicked"]["lat"])
        lon = float(res_map["last_clicked"]["lng"])
        st.session_state["spot_lat"] = lat
        st.session_state["spot_lon"] = lon
    st.markdown(f"**✓ Selected: lat={lat:.6f}, lon={lon:.6f}**")
else:
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))
    st.markdown(f"**✓ Selected: lat={lat:.6f}, lon={lon:.6f}**")
    
# 2) Pricing/inputs
st.markdown("#### 2) Enter free minutes / average stay / hourly price / capacity")
colA, colB, colC, colD = st.columns(4)
with colA:
    free_minutes = st.number_input("Free minutes", min_value=0.0, value=15.0, step=5.0)
with colB:
    avg_stay_hours = st.number_input("Average stay (hours)", min_value=0.0, value=2.0, step=0.25, format="%.2f")
with colC:
    price_per_hour = st.number_input("Price per hour (currency/hour)", min_value=0.0, value=20.0, step=1.0, format="%.2f")
with colD:
    capacity = st.number_input("Capacity (vehicles)", min_value=0.0, value=100.0, step=1.0, format="%.0f")

model_names = sorted(models.keys()) if models else []
if not model_names:
    st.error("No model JSON found. Place files in 06_result/.")

btn = st.button("Run prediction for this location")

# Run
if btn:
    # 1) OSM read & visualize (building categories)
    if osm_gdf is None:
        st.error("OSM GPKG not found.")
        st.stop()
    if gpd is None or Point is None:
        st.error("geopandas/shapely is required. Install them and retry.")
        st.stop()
    if not model_names:
        st.error("No model JSON found. Place files in 06_result/.")
        st.stop()

    gdf_wgs = osm_gdf

    # 500m counts (Commercial_500 and total)
    cnts = count_buildings_within_500m(gdf_wgs, lat, lon, building_filter=None)
    st.success(f"Buildings within 500m: total_500={cnts['total_500']}, Commercial_500={cnts['Commercial_500']}")

    # Draw map colored by building type
    draw_map_with_buildings(gdf_wgs, lat, lon)

    # 2) Model JSON (server bundled; pick the first one)
    bundle = models[model_names[0]]
    models, eval_pack, metrics_full = normalize_models_bundle(bundle)

    # 3) Build feature row (missing -> 0)
    row = {
        "Commercial_500": float(cnts["Commercial_500"]),
        "free_minutes": float(free_minutes),
        "price_per_hour": float(price_per_hour),
        "Capacity (lot)": float(capacity),
        "lat": float(lat), "lon": float(lon)
    }
    df_one = pd.DataFrame([row])

    # 4) Predict: choose aveHours_* target
    formula = None
    for m in models:
        fm = m.get("formula", {})
        tgt = fm.get("target", "")
        if "aveHours" in tgt or "avehours" in tgt.lower():
            formula = fm
            break
    if formula is None:
        st.error("No aveHours_* target found in the model JSON.")
        st.stop()

    feats = formula.get("features", [])
    X = prepare_X_for_formula(df_one, feats)
    yhat = predict_formula(X, formula)
    pred_hours = float(yhat.iloc[0])

    # （参考）CI：RMSE/MAPE が評価JSONにあれば 06_predict_apply と同じ帯を出す
    rmse, mape = rmse_mape_from_eval(eval_pack, metrics_full)
    z = 1.0  # 68% 相当（簡易）。必要に応じて変更可。
    y_lo_rmse = pred_hours - z * rmse
    y_hi_rmse = pred_hours + z * rmse
    y_lo_rel  = pred_hours * (1 - z * mape)
    y_hi_rel  = pred_hours * (1 + z * mape)
    pred_hours_lo = max(0.0, max(y_lo_rmse, y_lo_rel))
    pred_hours_hi = max(pred_hours_lo, min(y_hi_rmse, y_hi_rel))

       # 5) Vehicles and revenue (B plan: exponential -> mean diff)
    eps = 1e-6
    stay = max(float(avg_stay_hours), eps)
    counts = pred_hours / stay

    paid_per_car = paid_per_car_exponential(avg_stay_hours, free_minutes)
    if paid_per_car <= 0.0:
        paid_per_car = paid_per_car_meandiff(avg_stay_hours, free_minutes)

    revenue = counts * price_per_hour * paid_per_car

    # Display
    st.markdown("### Prediction Results")
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pred person-hours/day", f"{pred_hours:,.2f}")
    c2.metric("Pred vehicles/day", f"{counts:,.2f}")
    c3.metric("Paid hours per vehicle (expected)", f"{paid_per_car:,.2f}")
    c4.metric("Daily revenue (paid)", f"{revenue:,.2f}")

    with st.expander("Inputs and derived features"):
        st.json({
            "lat": lat, "lon": lon,
            "Commercial_500": cnts["Commercial_500"],
            "free_minutes": free_minutes,
            "avg_stay_hours": avg_stay_hours,
            "price_per_hour": price_per_hour,
            "capacity": capacity
        })

    # Save to session state (persist on rerun)
    st.session_state["_spot"] = {
        "lat": lat,
        "lon": lon,
        "last_result": {
            "pred_hours": pred_hours,
            "counts": counts,
            "paid_per_car": paid_per_car,
            "revenue": revenue,
            "rmse": rmse,
            "mape": mape,
            "pred_hours_lo": pred_hours_lo,
            "pred_hours_hi": pred_hours_hi
        }
    }

# ===== Page footer: auto re-display previous results =====
if "_spot" in st.session_state:
    sr = st.session_state["_spot"]
    # Re-display map
    if osm_gdf is not None and sr.get("lat") is not None and sr.get("lon") is not None:
        st.markdown("---")
        st.markdown("### Previous Map (auto re-display)")
        try:
            draw_map_with_buildings(osm_gdf, float(sr["lat"]), float(sr["lon"]))
        except Exception as e:
            st.info(f"Could not re-display the previous map: {e}")

    # Re-display metrics
    if sr.get("last_result") is not None:
        st.markdown("### Previous Prediction Results (auto re-display)")
        r = sr["last_result"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pred person-hours/day", f"{r.get('pred_hours', 0.0):,.2f}")
        c2.metric("Pred vehicles/day", f"{r.get('counts', 0.0):,.2f}")
        c3.metric("Paid hours per vehicle (expected)", f"{r.get('paid_per_car', 0.0):,.2f}")
        c4.metric("Daily revenue (paid)", f"{r.get('revenue', 0.0):,.2f}")

#        with st.expander("Reference: person-hours range (+/-1 sigma & +/-MAPE)"):
#            st.json({
#                "RMSE": r.get("rmse", 0.0),
#                "MAPE(0-1)": r.get("mape", 0.0),
#                "Pred_hours_lo": r.get("pred_hours_lo", 0.0),
#                "Pred_hours_hi": r.get("pred_hours_hi", 0.0),
#            })