# main.py
import os, io, zipfile, re, json
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# 지도용
import geopandas as gpd
from shapely.geometry import shape

st.set_page_config(page_title="서울시 상권 분기 매출 지도 분석", layout="wide")

# ---- 고정: 원본 열이름 그대로 사용 (해석은 '분기')
QUARTER_COL = "기준_년분기_코드"
AMT_COL     = "당월_매출_금액"   # 분기금액으로 해석하여 표시
CNT_COL     = "당월_매출_건수"

MILLION = 1_000_000  # 백만원 환산

# ---- 유틸
def to_num(s):
    return pd.to_numeric(pd.Series(s, dtype="object").astype(str).str.replace(",", "", regex=False), errors="coerce")

def parse_yq(s: str):
    s = str(s)
    m = re.search(r"(20\d{2}).*?([1-4])", s)
    if m: return int(m.group(1)), int(m.group(2))
    if s.isdigit() and len(s) in (5,6): return int(s[:4]), int(s[-1])
    return (9999, 9)

def pct(a, b):
    return (a/b - 1) * 100 if (b and b != 0) else np.nan

def read_csv_any(file):
    for enc in ["utf-8-sig","cp949","euc-kr"]:
        try:
            df = pd.read_csv(file, encoding=enc)
            df.columns = df.columns.map(lambda c: str(c).strip().replace("\u200b",""))
            return df
        except Exception:
            file.seek(0) if hasattr(file, "seek") else None
            continue
    return None

def read_xls_header2(file_or_path):
    df = pd.read_excel(file_or_path, sheet_name=0, header=2)
    df.columns = df.columns.map(lambda c: str(c).strip().replace("\u200b",""))
    return df

def load_geo_from_zip_or_geojson(uploaded):
    # uploaded: BytesIO/UploadedFile
    name = uploaded.name.lower()
    if name.endswith(".geojson") or name.endswith(".json"):
        gj = json.load(uploaded)
        gdf = gpd.GeoDataFrame.from_features(gj["features"])
        return gdf
    elif name.endswith(".zip"):
        # shapefile zip
        mem = io.BytesIO(uploaded.read())
        with zipfile.ZipFile(mem) as zf:
            tmpdir = "./_shp_tmp"
            os.makedirs(tmpdir, exist_ok=True)
            zf.extractall(tmpdir)
            # find .shp
            shp_paths = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.lower().endswith(".shp")]
            if not shp_paths:
                raise RuntimeError("zip 안에 .shp 파일이 없습니다.")
            gdf = gpd.read_file(shp_paths[0])
            return gdf
    else:
        raise RuntimeError("지원하는 지도 형식은 .geojson/.json 또는 .zip(shapefile) 입니다.")

# ---- 업로드
st.title("🗺️ 서울시 상권 분기 매출 지도 분석 (단위: 백만원)")

c1, c2 = st.columns([1,1])
with c1:
    sales_file = st.file_uploader("① 매출 CSV 업로드", type=["csv"])
    attr_file  = st.file_uploader("② 상권 속성 XLS 업로드", type=["xls","xlsx"])
with c2:
    geo_file   = st.file_uploader("③ 행정동 경계 업로드 (.geojson/.json 또는 .zip(shp))", type=["geojson","json","zip"])
    code_file  = st.file_uploader("④ (선택) 행정구역 코드표 XLS (읍면동 코드/명칭 표준화)", type=["xls","xlsx"])

if not (sales_file and attr_file and geo_file):
    st.info("세 파일(①②③)을 모두 업로드하면 분석이 시작됩니다.")
    st.stop()

# ---- 로딩
sales = read_csv_any(sales_file)
if sales is None or sales.empty:
    st.error("매출 CSV를 읽지 못했습니다."); st.stop()

attr = read_xls_header2(attr_file)
geo = load_geo_from_zip_or_geojson(geo_file)

# 컬럼 정리
for df in (sales, attr, geo):
    df.columns = df.columns.map(lambda c: str(c).strip().replace("\u200b",""))

# 필수 컬럼 검사
need_sales = [QUARTER_COL, AMT_COL, CNT_COL]
miss = [c for c in need_sales if c not in sales.columns]
if miss: st.error(f"매출 CSV 필수 컬럼 없음: {miss}"); st.stop()

need_attr = ["상권코드","상권명","행정동","자치구","면적(㎡)"]
miss2 = [c for c in need_attr if c not in attr.columns]
if miss2: st.error(f"상권 속성 XLS 필수 컬럼 없음: {miss2}"); st.stop()

# 행정동 경계 속성 후보 파악
adm_cd_col = None; adm_nm_col = None
for c in geo.columns:
    if c.upper() in ("ADM_CD","EMD_CD","DONG_CD","법정동코드","법정동코드10","행정동코드"): adm_cd_col = c
    if c.upper() in ("ADM_NM","EMD_NM","DONG_NM","법정동명","행정동명"):       adm_nm_col = c
if adm_cd_col is None and adm_nm_col is None:
    st.error("경계 파일에서 행정동 코드/명 컬럼을 찾지 못했습니다. 속성 열을 확인해주세요.")
    st.stop()

# (선택) 코드표 로드 → 읍면동명⇄코드 보정
if code_file:
    code = pd.read_excel(code_file, sheet_name=0, header=1)
    code.columns = ["시도코드","시도명칭","시군구코드","시군구명칭","읍면동코드","읍면동명칭"]
    code["읍면동코드"] = code["읍면동코드"].astype(str).str.zfill(10)
else:
    code = None

# ---- 매출↔상권속성 병합 (상권 코드/명)
# 매출데이터 쪽 상권 키 후보: 상권_코드 또는 상권_코드_명 / 상권_코드_명(이름)
sales_keys = [c for c in sales.columns if "상권" in c and ("코드" in c or "명" in c)]
attr_keys  = ["상권코드","상권명"]

# 우선 코드로, 없으면 명칭으로 병합
merge_on_sales = None; merge_on_attr = None
if any("상권_코드" == c for c in sales.columns):
    merge_on_sales = "상권_코드"; merge_on_attr = "상권코드"
elif any("상권_코드" in c for c in sales.columns):
    # 가장 근접한 열 하나
    merge_on_sales = [c for c in sales.columns if "상권_코드" in c][0]; merge_on_attr="상권코드"
elif any("상권_코드_명" == c for c in sales.columns):
    merge_on_sales = "상권_코드_명"; merge_on_attr = "상권명"
elif any("상권_코드_명" in c for c in sales.columns):
    merge_on_sales = [c for c in sales.columns if "상권_코드_명" in c][0]; merge_on_attr="상권명"
else:
    st.error(f"매출 CSV에서 상권 키를 찾지 못했습니다. 후보 열: {sales_keys}")
    st.stop()

merged = sales.merge(attr, left_on=merge_on_sales, right_on=merge_on_attr, how="left")

# ---- 행정동 매핑 (행정동명 → 코드, 또는 코드 직접)
if "행정동" not in merged.columns and "행정동명" in merged.columns:
    merged = merged.rename(columns={"행정동명":"행정동"})

if code is not None and ("읍면동코드" in code.columns):
    # 행정동명 기준으로 코드 매핑
    merged = merged.merge(code[["읍면동코드","읍면동명칭"]],
                          left_on="행정동", right_on="읍면동명칭", how="left")
    merged["행정동코드"] = merged["읍면동코드"].astype(str).str.zfill(10)
else:
    # 코드표 없으면 명칭으로만 조인할 준비
    merged["행정동코드"] = np.nan

# ---- 사이드바 필터
with st.sidebar:
    st.header("🔎 필터")
    uniq_q = sorted(merged[QUARTER_COL].astype(str).unique(), key=parse_yq)
    default_last = uniq_q[-5:] if len(uniq_q)>=5 else uniq_q
    picked = st.multiselect("분기 선택", uniq_q, default_last)

    # 자치구/상권분류 필터 (있을 때)
    chosen = []
    for col in ["자치구","상권분류"]:
        if col in merged.columns:
            vals = sorted(merged[col].dropna().astype(str).unique())
            sel = st.multiselect(f"{col} 선택", vals)
            if sel: chosen.append((col, sel))

    st.divider()
    TOPN = st.slider("Top-N", 5, 50, 20, 1)

work = merged.copy()
if picked:
    work = work[work[QUARTER_COL].astype(str).isin(picked)]
for c,vals in chosen:
    work = work[work[c].astype(str).isin(vals)]
if work.empty:
    st.warning("필터 결과가 없습니다."); st.stop()

# ---- KPI (백만원)
st.subheader("📌 요약 (단위: 백만원)")
sum_amt_m = to_num(work[AMT_COL]).sum() / MILLION
sum_cnt   = to_num(work[CNT_COL]).sum()
avg_price_m = (sum_amt_m / sum_cnt) if sum_cnt else np.nan
c1,c2,c3 = st.columns(3)
c1.metric("총 매출(백만원)", f"{sum_amt_m:,.1f}")
c2.metric("총 건수", f"{sum_cnt:,.0f} 건")
c3.metric("평균 객단가(백만원/건)", f"{avg_price_m:,.3f}" if pd.notna(avg_price_m) else "N/A")

# ---- 분기별 매출 추이 (막대, 백만원)
st.subheader("📈 분기별 매출 추이 (막대, 단위: 백만원)")
trend = (
    work.groupby(QUARTER_COL)[AMT_COL]
        .agg(lambda s: to_num(s).sum())
        .reset_index(name="매출 금액(백만원)")
        .sort_values(by=QUARTER_COL, key=lambda s: s.astype(str).map(parse_yq))
)
trend["매출 금액(백만원)"] = trend["매출 금액(백만원)"] / MILLION
st.altair_chart(
    alt.Chart(trend).mark_bar().encode(
        x=alt.X(f"{QUARTER_COL}:N", title="분기"),
        y=alt.Y("매출 금액(백만원):Q", title="매출 금액(백만원)"),
        tooltip=[QUARTER_COL, alt.Tooltip("매출 금액(백만원):Q", format=",.1f")]
    ),
    use_container_width=True
)

# ---- 행정동 단위 집계 (지도용)
# 1) 행정동코드가 있으면 코드 기준, 없으면 명칭 기준으로 geo와 조인
geo = geo.copy()
if adm_cd_col:
    # shapefile 코드 문자열 통일
    geo[adm_cd_col] = geo[adm_cd_col].astype(str).str.zfill(10)
    if "행정동코드" in work.columns and work["행정동코드"].notna().any():
        agg = (
            work.groupby("행정동코드")[AMT_COL]
                .agg(lambda s: to_num(s).sum())
                .reset_index(name="매출(백만원)")
        )
        agg["매출(백만원)"] = agg["매출(백만원)"] / MILLION
        joined = geo.merge(agg, left_on=adm_cd_col, right_on="행정동코드", how="left")
    else:
        # 명칭 기준
        base_nm = adm_nm_col if adm_nm_col else adm_cd_col
        agg = (
            work.groupby("행정동")[AMT_COL]
                .agg(lambda s: to_num(s).sum())
                .reset_index(name="매출(백만원)")
        )
        agg["매출(백만원)"] = agg["매출(백만원)"] / MILLION
        if adm_nm_col:
            joined = geo.merge(agg, left_on=adm_nm_col, right_on="행정동", how="left")
        else:
            st.warning("경계에 행정동명 컬럼이 없어 명칭 기준 조인이 어렵습니다. 코드표 업로드를 권장합니다.")
            joined = geo.copy()
else:
    # 코드 컬럼이 전혀 없을 때: 명칭 기준만 시도
    agg = (
        work.groupby("행정동")[AMT_COL]
            .agg(lambda s: to_num(s).sum())
            .reset_index(name="매출(백만원)")
    )
    agg["매출(백만원)"] = agg["매출(백만원)"] / MILLION
    if adm_nm_col:
        joined = geo.merge(agg, left_on=adm_nm_col, right_on="행정동", how="left")
    else:
        st.error("경계에서 코드/명칭 컬럼을 찾지 못했습니다."); st.stop()

# ---- 지도 (pydeck: Choropleth)
st.subheader("🗺️ 행정동 매출 Choropleth (단위: 백만원)")
# 중심 추정
try:
    center = joined.geometry.unary_union.centroid
    view_state = {"latitude": center.y, "longitude": center.x, "zoom": 10}
except Exception:
    view_state = {"latitude": 37.5665, "longitude": 126.9780, "zoom": 10}

# pydeck은 GeoJSON이 편해 변환
gj = json.loads(joined.to_json())

import pydeck as pdk
layer = pdk.Layer(
    "GeoJsonLayer",
    gj,
    opacity=0.6,
    stroked=True,
    get_fill_color="[min(255, 80 + (properties['매출(백만원)'] || 0) * 2), 120, 180]",
    get_line_color=[50, 50, 50],
    line_width_min_pixels=1,
    pickable=True,
)
tooltip = {"html": "<b>{properties.ADM_NM}</b><br/>매출: {properties.매출(백만원)} 백만원", "style": {"color": "white"}}

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(**view_state), tooltip=tooltip))

# ---- 랭킹 (행정동/상권/상권분류 Top-N)
st.subheader(f"🏆 Top {TOPN}")
col_opts = []
if "행정동" in work.columns: col_opts.append("행정동")
if "상권명" in work.columns: col_opts.append("상권명")
if "상권분류" in work.columns: col_opts.append("상권분류")

if col_opts:
    grp_col = st.selectbox("랭킹 기준 컬럼", col_opts, index=0)
    rk = (
        work.groupby(grp_col)[AMT_COL]
            .agg(lambda s: to_num(s).sum())
            .reset_index(name="매출(백만원)")
            .sort_values("매출(백만원)", ascending=False)
            .head(TOPN)
    )
    rk["매출(백만원)"] = rk["매출(백만원)"] / MILLION
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(rk, use_container_width=True)
    with c2:
        st.altair_chart(
            alt.Chart(rk).mark_bar().encode(
                x=alt.X("매출(백만원):Q", title="매출(백만원)"),
                y=alt.Y(f"{grp_col}:N", sort="-x"),
                tooltip=[grp_col, alt.Tooltip("매출(백만원):Q", format=",.1f")]
            ), use_container_width=True
        )
else:
    st.info("랭킹을 계산할 적절한 열을 찾지 못했습니다.")

# ---- 다운로드
st.subheader("📥 지도 조인 결과 다운로드")
st.download_button(
    "행정동 조인 GeoJSON 다운로드",
    data=json.dumps(gj, ensure_ascii=False),
    file_name="joined_admdong_sales.geojson",
    mime="application/geo+json"
)
