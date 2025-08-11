# main.py
import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="서울시 상권분석 분기 매출 대시보드", layout="wide")

# ───────────────────────────
# 유틸
# ───────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(file_like_or_path):
    encodings = ["utf-8-sig", "cp949", "euc-kr"]
    for enc in encodings:
        try:
            return pd.read_csv(file_like_or_path, encoding=enc)
        except:
            continue
    st.error("CSV 파일을 읽을 수 없습니다.")
    return None

def to_num(s):
    return pd.to_numeric(
        pd.Series(s, dtype="object").astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

def find_col(df, patterns):
    for col in df.columns:
        for pat in patterns:
            if re.search(pat, str(col)):
                return col
    return None

def parse_yq(x):
    s = str(x)
    m = re.search(r"(20\d{2}).*?([1-4])", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    if s.isdigit() and len(s) in (5,6):
        return int(s[:4]), int(s[-1])
    return (9999, 9)

# ───────────────────────────
# 데이터 로드
# ───────────────────────────
uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"])
if uploaded:
    df = load_csv(uploaded)
else:
    st.info("샘플 데이터를 로드합니다. (같은 경로의 CSV)")
    sample_path = os.path.join(os.path.dirname(__file__), "서울시 상권분석서비스(추정매출-상권)_sample.csv")
    if os.path.exists(sample_path):
        df = load_csv(sample_path)
    else:
        st.error("샘플 CSV가 없습니다.")
        st.stop()

if df is None:
    st.stop()

df.columns = df.columns.map(lambda c: str(c).strip())

# ───────────────────────────
# 핵심 컬럼 탐지
# ───────────────────────────
col_quarter = find_col(df, [r"기준.?년분기.?코드"])
col_amt = find_col(df, [r"분기.?매출.?금액", r"매출.?금액"])
col_cnt = find_col(df, [r"분기.?매출.?건수", r"매출.?건수"])

if not col_quarter or not col_amt or not col_cnt:
    st.error("분기, 매출금액, 매출건수 컬럼을 찾을 수 없습니다.")
    st.stop()

# ───────────────────────────
# 사이드바 필터
# ───────────────────────────
with st.sidebar:
    st.header("🔍 필터")
    chosen = {}
    for col in ["상권_코드_명", "서비스_업종_코드_명"]:
        if col in df.columns:
            vals = sorted(df[col].dropna().astype(str).unique())
            sel = st.multiselect(f"{col} 선택", vals)
            if sel:
                chosen[col] = sel

    st.divider()
    metric_col = st.radio("랭킹 지표", ["분기 매출 금액", "분기 매출 건수"], index=0)
    topn = st.slider("Top-N", 5, 30, 10, 1)

for col, vals in chosen.items():
    df = df[df[col].astype(str).isin(vals)]

if df.empty:
    st.warning("조건에 맞는 데이터가 없습니다.")
    st.stop()

# ───────────────────────────
# 요약 지표
# ───────────────────────────
st.subheader("📌 분기별 요약 지표")
amt_sum = to_num(df[col_amt]).sum()
cnt_sum = to_num(df[col_cnt]).sum()
avg_price = amt_sum / cnt_sum if cnt_sum else None

c1, c2, c3 = st.columns(3)
c1.metric("총 분기 매출 금액", f"{amt_sum:,.0f} 원")
c2.metric("총 분기 매출 건수", f"{cnt_sum:,.0f} 건")
c3.metric("평균 객단가", f"{avg_price:,.0f} 원" if avg_price else "계산 불가")

# ───────────────────────────
# 분기별 매출 추이
# ───────────────────────────
st.subheader("📈 분기별 매출 추이")
trend = (
    df.groupby(col_quarter)[col_amt]
    .apply(lambda s: to_num(s).sum())
    .reset_index()
    .rename(columns={col_amt: "매출 금액"})
)
trend["__sort__"] = trend[col_quarter].map(parse_yq)
trend = trend.sort_values("__sort__").drop(columns="__sort__")
st.altair_chart(
    alt.Chart(trend).mark_line(point=True).encode(
        x=alt.X(f"{col_quarter}:N", title="분기"),
        y=alt.Y("매출 금액:Q", title="매출 금액"),
        tooltip=[col_quarter, alt.Tooltip("매출 금액:Q", format=",")]
    ),
    use_container_width=True
)

# ───────────────────────────
# 랭킹
# ───────────────────────────
metric_actual = col_amt if metric_col == "분기 매출 금액" else col_cnt
st.subheader(f"🏆 {metric_col} 기준 Top {topn}")
if "서비스_업종_코드_명" in df.columns:
    top_df = (
        df.groupby("서비스_업종_코드_명")[metric_actual]
        .apply(lambda s: to_num(s).sum())
        .reset_index()
        .sort_values(by=metric_actual, ascending=False)
        .head(topn)
    )
    st.dataframe(top_df, use_container_width=True)
    st.altair_chart(
        alt.Chart(top_df).mark_bar().encode(
            x=alt.X(f"{metric_actual}:Q", title=metric_col),
            y=alt.Y("서비스_업종_코드_명:N", sort="-x"),
            tooltip=["서비스_업종_코드_명", alt.Tooltip(metric_actual, format=",")]
        ),
        use_container_width=True
    )

# ───────────────────────────
# 데이터 테이블 + 다운로드
# ───────────────────────────
st.subheader("📥 필터 반영 데이터")
st.dataframe(df, use_container_width=True)
st.download_button(
    "CSV 다운로드",
    data=df.to_csv(index=False, encoding="utf-8-sig"),
    file_name="filtered_data.csv",
    mime="text/csv"
)
