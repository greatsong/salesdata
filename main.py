# main.py
import os, re
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="서울시 상권 분기 매출 대시보드", layout="wide")

# ── 고정: 파일의 '정확한' 열 이름 사용(이름은 그대로, 해석은 '분기')
QUARTER_COL = "기준_년분기_코드"
AMT_COL     = "당월_매출_금액"   # 파일 명칭 그대로 사용하지만 '분기 매출'로 해석/표시
CNT_COL     = "당월_매출_건수"

# ── 유틸
@st.cache_data(show_spinner=False)
def load_csv(file_like_or_path):
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(file_like_or_path, encoding=enc)
        except Exception:
            continue
    return None

def to_num(s):
    return pd.to_numeric(pd.Series(s, dtype="object").astype(str).str.replace(",", "", regex=False),
                         errors="coerce")

def parse_yq(s: str):
    s = str(s)
    m = re.search(r"(20\d{2}).*?([1-4])", s)
    if m: return int(m.group(1)), int(m.group(2))
    if s.isdigit() and len(s) in (5,6): return int(s[:4]), int(s[-1])
    return (9999, 9)

def prev_yq_str(y, q):
    py, pq = (y-1, 4) if q == 1 else (y, q-1)
    return f"{py}Q{pq}"

def yoy_yq_str(y, q):
    return f"{y-1}Q{q}"

def pct(a, b):
    return (a/b - 1) * 100 if (b and b != 0) else np.nan

MILLION = 1_000_000  # 백만원 환산

# ── 데이터 선택: 업로드 없으면 현재 폴더 CSV 목록에서 선택(한글 경로 이슈 회피)
st.title("📊 서울시 상권 **분기** 매출 대시보드 (단위: 백만원)")

up = st.file_uploader("CSV 업로드 (없으면 아래에서 선택)", type=["csv"])
df = None
src_desc = ""

if up is not None:
    df = load_csv(up)
    src_desc = "업로드 파일"
else:
    csvs = [f for f in os.listdir(".") if f.lower().endswith(".csv")]
    if not csvs:
        st.error("현재 폴더에 CSV가 없습니다. 파일을 업로드하세요.")
        st.stop()
    sel = st.selectbox("현재 폴더의 CSV 중 선택", options=csvs, index=0)
    if sel:
        df = load_csv(sel)
        src_desc = f"로컬 파일: {sel}"

if df is None or df.empty:
    st.error("CSV를 읽지 못했습니다. 인코딩/파일을 확인하세요.")
    st.stop()

df.columns = df.columns.map(lambda c: str(c).strip().replace("\u200b",""))

missing = [c for c in [QUARTER_COL, AMT_COL, CNT_COL] if c not in df.columns]
if missing:
    st.error(f"필수 컬럼이 없습니다: {missing}\n\n현재 컬럼 목록: {list(df.columns)}")
    st.stop()

st.caption(f"데이터 소스: {src_desc}")

# ── 사이드바 필터
with st.sidebar:
    st.header("🔎 필터")
    uniq_q = sorted(df[QUARTER_COL].astype(str).unique(), key=parse_yq)
    default_last = uniq_q[-5:] if len(uniq_q) >= 5 else uniq_q
    picked = st.multiselect("분기 선택", options=uniq_q, default=default_last)

    extra_filters = []
    for col in ["상권_코드_명", "서비스_업종_코드_명", "자치구", "상권_코드"]:
        if col in df.columns:
            vals = sorted(df[col].dropna().astype(str).unique())
            sel = st.multiselect(f"{col} 선택", vals)
            if sel:
                extra_filters.append((col, sel))

    st.divider()
    metric_name = st.radio("지표", ["매출 금액(백만원)", "매출 건수"], index=0)
    TOPN = st.slider("Top-N", 5, 50, 20, 1)

work = df.copy()
if picked:
    work = work[work[QUARTER_COL].astype(str).isin(picked)]
for col, vals in extra_filters:
    work = work[work[col].astype(str).isin(vals)]

if work.empty:
    st.warning("필터 결과가 없습니다. 조건을 완화하세요.")
    st.stop()

METRIC_COL = AMT_COL if metric_name.startswith("매출 금액") else CNT_COL
is_amt = METRIC_COL == AMT_COL

# ── KPI (단위: 백만원)
st.subheader("📌 요약 지표 (선택 분기 합계, 단위: 백만원)")
sum_amt_m = to_num(work[AMT_COL]).sum() / MILLION
sum_cnt = to_num(work[CNT_COL]).sum()
avg_price_m = (sum_amt_m / sum_cnt) if sum_cnt else np.nan  # 건당 백만원

c1, c2, c3 = st.columns(3)
c1.metric("총 매출 금액(백만원)", f"{sum_amt_m:,.1f}")
c2.metric("총 매출 건수", f"{sum_cnt:,.0f} 건")
c3.metric("평균 객단가(백만원/건)", f"{avg_price_m:,.3f}" if pd.notna(avg_price_m) else "계산 불가")

# 최근분기 QoQ/YoY (금액은 백만원 단위로 계산/표시)
uniq_q_sorted = sorted(work[QUARTER_COL].astype(str).unique(), key=parse_yq)
last_q = uniq_q_sorted[-1]
last_y, last_qu = parse_yq(last_q)
prev_q = prev_yq_str(last_y, last_qu)
yoy_q  = yoy_yq_str(last_y, last_qu)

def sum_for_million(q, col):
    s = work[work[QUARTER_COL].astype(str) == q]
    val = to_num(s[col]).sum()
    return (val / MILLION) if col == AMT_COL else val

cur_m = sum_for_million(last_q, AMT_COL)
prv_m = sum_for_million(prev_q, AMT_COL) if prev_q in uniq_q_sorted else np.nan
yy_m  = sum_for_million(yoy_q,  AMT_COL) if yoy_q  in uniq_q_sorted else np.nan

c4, c5, c6 = st.columns(3)
c4.metric(f"{last_q} 매출(백만원)", f"{cur_m:,.1f}")
c5.metric("QoQ", f"{pct(cur_m, prv_m):.1f} %" if pd.notna(prv_m) else "N/A")
c6.metric("YoY", f"{pct(cur_m, yy_m):.1f} %" if pd.notna(yy_m) else "N/A")

# ── 분기별 매출 추이: ✔ 막대그래프, ✔ 단위=백만원, ✔ 열이름에 단위 명시
st.subheader("📈 분기별 매출 추이 (단위: 백만원)")
trend = (
    work.groupby(QUARTER_COL)[AMT_COL]
        .agg(lambda s: to_num(s).sum())
        .reset_index(name="매출 금액(백만원)")
        .sort_values(by=QUARTER_COL, key=lambda s: s.astype(str).map(parse_yq))
)
trend["매출 금액(백만원)"] = trend["매출 금액(백만원)"] / MILLION

st.altair_chart(
    alt.Chart(trend).mark_bar().encode(  # ← 막대그래프
        x=alt.X(f"{QUARTER_COL}:N", title="분기"),
        y=alt.Y("매출 금액(백만원):Q", title="매출 금액(백만원)"),
        tooltip=[QUARTER_COL, alt.Tooltip("매출 금액(백만원):Q", format=",.1f")]
    ),
    use_container_width=True
)

# ── 랭킹 (업종/상권): 금액은 백만원으로 환산해 표/축/열이름에 (백만원) 명시
st.subheader(f"🏆 {metric_name} 기준 Top {TOPN}")

if "서비스_업종_코드_명" in work.columns:
    # 그룹 합계
    up = (
        work.groupby("서비스_업종_코드_명")[METRIC_COL]
            .agg(lambda s: to_num(s).sum())
            .reset_index(name="값")
    )
    if is_amt:
        up["값(백만원)"] = up["값"] / MILLION
        col_for_chart = "값(백만원)"
        title_for_chart = "매출 금액(백만원)"
    else:
        col_for_chart = "값"
        title_for_chart = "매출 건수"

    up = up.sort_values(col_for_chart, ascending=False).head(TOPN)
    st.caption("업종 Top-N")
    st.dataframe(up[[ "서비스_업종_코드_명", col_for_chart ]].rename(columns={col_for_chart: title_for_chart}),
                 use_container_width=True)
    st.altair_chart(
        alt.Chart(up).mark_bar().encode(
            x=alt.X(f"{col_for_chart}:Q", title=title_for_chart),
            y=alt.Y("서비스_업종_코드_명:N", sort="-x"),
            tooltip=["서비스_업종_코드_명", alt.Tooltip(col_for_chart, format=",.1f" if is_amt else ",.0f")]
        ),
        use_container_width=True
    )

if "상권_코드_명" in work.columns:
    ar = (
        work.groupby("상권_코드_명")[METRIC_COL]
            .agg(lambda s: to_num(s).sum())
            .reset_index(name="값")
    )
    if is_amt:
        ar["값(백만원)"] = ar["값"] / MILLION
        col_for_chart = "값(백만원)"
        title_for_chart = "매출 금액(백만원)"
    else:
        col_for_chart = "값"
        title_for_chart = "매출 건수"

    ar = ar.sort_values(col_for_chart, ascending=False).head(TOPN)
    st.caption("상권 Top-N")
    st.dataframe(ar[[ "상권_코드_명", col_for_chart ]].rename(columns={col_for_chart: title_for_chart}),
                 use_container_width=True)
    st.altair_chart(
        alt.Chart(ar).mark_bar().encode(
            x=alt.X(f"{col_for_chart}:Q", title=title_for_chart),
            y=alt.Y("상권_코드_명:N", sort="-x"),
            tooltip=["상권_코드_명", alt.Tooltip(col_for_chart, format=",.1f" if is_amt else ",.0f")]
        ),
        use_container_width=True
    )

# ── 현재 데이터 + 다운로드(원본 그대로)
st.subheader("📥 현재 필터 반영 데이터 (원본 컬럼)")
st.dataframe(work, use_container_width=True)
st.download_button(
    "CSV 다운로드(원본)",
    data=work.to_csv(index=False, encoding="utf-8-sig"),
    file_name="filtered_data.csv",
    mime="text/csv"
)
