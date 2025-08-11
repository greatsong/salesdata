# main.py
import os, re
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="서울시 상권 분기 매출 대시보드", layout="wide")

# ====== 고정: 파일의 '정확한' 열 이름 사용 ======
QUARTER_COL = "기준_년분기_코드"
AMT_COL     = "당월_매출_금액"   # 파일 상 표기가 '당월'이어도 본 앱에서는 '분기 합계'로 취급
CNT_COL     = "당월_매출_건수"

# ====== 유틸 ======
@st.cache_data(show_spinner=False)
def load_csv(file_like_or_path):
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(file_like_or_path, encoding=enc)
        except Exception:
            continue
    return None

def to_num(s):
    """천단위 콤마 등 문자열도 안전하게 숫자로 변환"""
    return pd.to_numeric(
        pd.Series(s, dtype="object").astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

def parse_yq(s: str):
    """'2024Q1', '2024 1', '20241' 등에서 (연,분기) 튜플 반환"""
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

# ====== 데이터 소스 선택 ======
st.title("📊 서울시 상권 **분기** 매출 대시보드 (원본 열이름 그대로)")

up = st.file_uploader("CSV 업로드 (없으면 아래에서 선택)", type=["csv"])
df = None
src_desc = ""

if up is not None:
    df = load_csv(up)
    src_desc = "업로드 파일"
else:
    # 현재 폴더의 CSV 목록에서 선택 (한글 파일명 경로 이슈 회피)
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

# 컬럼 전처리: 앞뒤 공백/숨은문자 제거
df.columns = df.columns.map(lambda c: str(c).strip().replace("\u200b",""))

# 필수 컬럼 체크 (정확히 존재해야 함)
missing = [c for c in [QUARTER_COL, AMT_COL, CNT_COL] if c not in df.columns]
if missing:
    st.error(f"필수 컬럼이 없습니다: {missing}\n\n현재 컬럼 목록: {list(df.columns)}")
    st.stop()

st.caption(f"데이터 소스: {src_desc}")

# ====== 사이드바 필터 ======
with st.sidebar:
    st.header("🔎 필터")
    # 최근 N개 분기 선택
    uniq_q = sorted(df[QUARTER_COL].astype(str).unique(), key=parse_yq)
    default_last = uniq_q[-5:] if len(uniq_q) >= 5 else uniq_q
    picked = st.multiselect("분기 선택", options=uniq_q, default=default_last)

    # 보조 필터(존재하는 경우만)
    extra_filters = []
    for col in ["상권_코드_명", "서비스_업종_코드_명", "자치구", "상권_코드"]:
        if col in df.columns:
            vals = sorted(df[col].dropna().astype(str).unique())
            sel = st.multiselect(f"{col} 선택", vals)
            if sel:
                extra_filters.append((col, sel))

    st.divider()
    metric_name = st.radio("지표", ["매출 금액", "매출 건수"], index=0)
    TOPN = st.slider("Top-N", 5, 50, 20, 1)

# 필터 적용
work = df.copy()
if picked:
    work = work[work[QUARTER_COL].astype(str).isin(picked)]
for col, vals in extra_filters:
    work = work[work[col].astype(str).isin(vals)]

if work.empty:
    st.warning("필터 결과가 없습니다. 조건을 완화하세요.")
    st.stop()

METRIC_COL = AMT_COL if metric_name == "매출 금액" else CNT_COL

# ====== KPI ======
st.subheader("📌 요약 지표 (선택 분기 합계)")
sum_amt = to_num(work[AMT_COL]).sum()
sum_cnt = to_num(work[CNT_COL]).sum()
avg_price = (sum_amt / sum_cnt) if sum_cnt else np.nan

c1, c2, c3 = st.columns(3)
c1.metric("총 매출 금액", f"{sum_amt:,.0f} 원")
c2.metric("총 매출 건수", f"{sum_cnt:,.0f} 건")
c3.metric("평균 객단가", f"{avg_price:,.0f} 원" if pd.notna(avg_price) else "계산 불가")

# 최근분기 / QoQ / YoY
uniq_q_sorted = sorted(work[QUARTER_COL].astype(str).unique(), key=parse_yq)
last_q = uniq_q_sorted[-1]
last_y, last_qu = parse_yq(last_q)
prev_q = prev_yq_str(last_y, last_qu)
yoy_q  = yoy_yq_str(last_y, last_qu)

def sum_for(q, col):
    s = work[work[QUARTER_COL].astype(str) == q]
    return to_num(s[col]).sum()

cur = sum_for(last_q, AMT_COL)
prv = sum_for(prev_q, AMT_COL) if prev_q in uniq_q_sorted else np.nan
yy  = sum_for(yoy_q, AMT_COL)  if yoy_q  in uniq_q_sorted else np.nan

c4, c5, c6 = st.columns(3)
c4.metric(f"{last_q} 매출", f"{cur:,.0f} 원")
c5.metric("QoQ", f"{pct(cur, prv):.1f} %" if pd.notna(prv) else "N/A")
c6.metric("YoY", f"{pct(cur, yy):.1f} %" if pd.notna(yy) else "N/A")

# ====== 분기별 매출 추이 (✔ 그룹 합계 후 reset_index) ======
st.subheader("📈 분기별 매출 추이")
trend = (
    work.groupby(QUARTER_COL)[AMT_COL]
        .agg(lambda s: to_num(s).sum())             # 그룹별 합
        .reset_index(name="매출 금액")               # 이름 지정하여 스칼라 오류 방지
        .sort_values(by=QUARTER_COL, key=lambda s: s.astype(str).map(parse_yq))
)
st.altair_chart(
    alt.Chart(trend).mark_line(point=True).encode(
        x=alt.X(f"{QUARTER_COL}:N", title="분기"),
        y=alt.Y("매출 금액:Q", title="매출 금액"),
        tooltip=[QUARTER_COL, alt.Tooltip("매출 금액:Q", format=",")]
    ),
    use_container_width=True
)

# ====== 랭킹 (업종/상권이 있으면 각각) — ✔ 그룹 합계 방식 통일 ======
st.subheader(f"🏆 {metric_name} 기준 Top {TOPN}")

if "서비스_업종_코드_명" in work.columns:
    up = (
        work.groupby("서비스_업종_코드_명")[METRIC_COL]
            .agg(lambda s: to_num(s).sum())
            .reset_index(name="값")
            .sort_values("값", ascending=False)
            .head(TOPN)
    )
    st.caption("업종 Top-N")
    st.dataframe(up, use_container_width=True)
    st.altair_chart(
        alt.Chart(up).mark_bar().encode(
            x=alt.X("값:Q", title=metric_name),
            y=alt.Y("서비스_업종_코드_명:N", sort="-x"),
            tooltip=["서비스_업종_코드_명", alt.Tooltip("값:Q", format=",")]
        ),
        use_container_width=True
    )

if "상권_코드_명" in work.columns:
    ar = (
        work.groupby("상권_코드_명")[METRIC_COL]
            .agg(lambda s: to_num(s).sum())
            .reset_index(name="값")
            .sort_values("값", ascending=False)
            .head(TOPN)
    )
    st.caption("상권 Top-N")
    st.dataframe(ar, use_container_width=True)
    st.altair_chart(
        alt.Chart(ar).mark_bar().encode(
            x=alt.X("값:Q", title=metric_name),
            y=alt.Y("상권_코드_명:N", sort="-x"),
            tooltip=["상권_코드_명", alt.Tooltip("값:Q", format=",")]
        ),
        use_container_width=True
    )

# ====== 현재 데이터 + 다운로드 ======
st.subheader("📥 현재 필터 반영 데이터")
st.dataframe(work, use_container_width=True)
st.download_button(
    "CSV 다운로드",
    data=work.to_csv(index=False, encoding="utf-8-sig"),
    file_name="filtered_data.csv",
    mime="text/csv"
)
