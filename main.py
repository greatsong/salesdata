# main.py
import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="서울시 상권분석 매출 대시보드", layout="wide")

# ──────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(file_like_or_path):
    """자동 인코딩 감지로 CSV 로드"""
    encodings = ["utf-8-sig", "cp949", "euc-kr"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(file_like_or_path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    st.error(f"CSV를 읽지 못했습니다. 인코딩을 확인하세요. 마지막 오류: {last_err}")
    return None

def to_num(s):
    """천단위 콤마가 섞여있어도 안전하게 숫자로 변환"""
    return pd.to_numeric(
        pd.Series(s, dtype="object").astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

def sum_by_cols(df, cols, label_cleaner=None):
    """여러 열을 세로형 합계 데이터프레임으로 변환"""
    if not cols:
        return pd.DataFrame()
    s = df[cols].apply(to_num).sum(axis=0)
    out = pd.DataFrame({"항목": s.index, "값": s.values})
    if label_cleaner:
        out["항목"] = out["항목"].map(label_cleaner)
    return out

def has_cols(df, cols):
    return all(c in df.columns for c in cols)

# ──────────────────────────────────────────────────────────
# 데이터 업로드/로드
# ──────────────────────────────────────────────────────────
st.title("📊 서울시 상권분석 매출 대시보드 (원본 포맷 그대로)")

uploaded = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
if uploaded:
    df = load_csv(uploaded)
else:
    st.info("샘플 데이터를 로드했습니다. (같은 경로에 있는 CSV)")
    sample_path = os.path.join(
        os.path.dirname(__file__),
        "서울시 상권분석서비스(추정매출-상권)_sample.csv"
    )
    if os.path.exists(sample_path):
        df = load_csv(sample_path)
    else:
        st.error(f"샘플 파일이 없습니다: {sample_path}")
        st.stop()

if df is None:
    st.stop()

# ──────────────────────────────────────────────────────────
# 사이드바 필터(존재하는 열만 자동 생성)
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 필터")
    filterable = []
    for c in ["상권_코드_명", "상권_코드", "서비스_업종_코드_명", "서비스_업종_코드", "기준_년분기_코드"]:
        if c in df.columns:
            filterable.append(c)

    chosen = {}
    for col in filterable:
        vals = sorted(df[col].dropna().astype(str).unique())
        sel = st.multiselect(f"{col} 선택", vals)
        if sel:
            chosen[col] = sel

    st.divider()
    metric_col = st.radio("랭킹 지표", ["당월_매출_금액", "당월_매출_건수"], index=0)
    topn = st.slider("Top-N (업종/상권)", 5, 30, 10, 1)

# 필터 적용
for col, vals in chosen.items():
    df = df[df[col].astype(str).isin(vals)]

if df.empty:
    st.warning("필터 결과가 없습니다. 조건을 완화해보세요.")
    st.stop()

# ──────────────────────────────────────────────────────────
# 상단 요약 카드
# ──────────────────────────────────────────────────────────
st.subheader("📌 요약 지표")
c1, c2, c3 = st.columns(3)
if has_cols(df, ["당월_매출_금액", "당월_매출_건수"]):
    total_sales = to_num(df["당월_매출_금액"]).sum()
    total_cnt = to_num(df["당월_매출_건수"]).sum()
    avg_price = (total_sales / total_cnt) if total_cnt else 0
    c1.metric("총 매출 금액", f"{total_sales:,.0f} 원")
    c2.metric("총 매출 건수", f"{total_cnt:,.0f} 건")
    c3.metric("평균 객단가", f"{avg_price:,.0f} 원")
else:
    c1.info("당월_매출_금액 없음")
    c2.info("당월_매출_건수 없음")
    c3.info("객단가 계산 불가")

# ──────────────────────────────────────────────────────────
# 기간별 매출 추이
# ──────────────────────────────────────────────────────────
if has_cols(df, ["기준_년분기_코드", "당월_매출_금액"]):
    st.subheader("📈 기간별 매출 추이")
    trend = (
        df.groupby("기준_년분기_코드")["당월_매출_금액"]
        .apply(lambda s: to_num(s).sum())
        .reset_index()
        .rename(columns={"당월_매출_금액": "매출 금액"})
        .sort_values("기준_년분기_코드")
    )
    chart = (
        alt.Chart(trend)
        .mark_line(point=True)
        .encode(
            x=alt.X("기준_년분기_코드:N", title="년분기"),
            y=alt.Y("매출 금액:Q", title="매출 금액"),
            tooltip=["기준_년분기_코드", alt.Tooltip("매출 금액:Q", format=",")]
        )
    )
    st.altair_chart(chart, use_container_width=True)

# ──────────────────────────────────────────────────────────
# 요일별 매출
# ──────────────────────────────────────────────────────────
weekday_order = ["월요일","화요일","수요일","목요일","금요일","토요일","일요일"]
weekday_cols_amt = [c for c in df.columns if c.endswith("_매출_금액") and any(w in c for w in weekday_order)]
if weekday_cols_amt:
    st.subheader("📅 요일별 매출 금액")
    def wk_clean(x): return x.replace("_매출_금액", "")
    wk = sum_by_cols(df, weekday_cols_amt, wk_clean)
    wk["sort"] = wk["항목"].map({w:i for i,w in enumerate(weekday_order)}).fillna(99)
    wk = wk.sort_values("sort").drop(columns="sort")
    bar = (
        alt.Chart(wk)
        .mark_bar()
        .encode(
            x=alt.X("항목:N", sort=list(wk["항목"]), title="요일"),
            y=alt.Y("값:Q", title="매출 금액"),
            tooltip=[alt.Tooltip("항목:N", title="요일"), alt.Tooltip("값:Q", title="매출 금액", format=",")]
        )
    )
    st.altair_chart(bar, use_container_width=True)

# ──────────────────────────────────────────────────────────
# 시간대별 매출
# ──────────────────────────────────────────────────────────
time_cols_amt = [c for c in df.columns if ("시간대" in c and c.endswith("_매출_금액"))]
if time_cols_amt:
    st.subheader("⏰ 시간대별 매출 금액")
    def tz_clean(x): return x.replace("시간대", "").replace("_매출_금액", "").strip("_")
    tz = sum_by_cols(df, time_cols_amt, tz_clean)

    def tz_key(s):
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else 999
    tz = tz.sort_values(by="항목", key=lambda s: s.map(tz_key))

    line = (
        alt.Chart(tz)
        .mark_line(point=True)
        .encode(
            x=alt.X("항목:N", title="시간대"),
            y=alt.Y("값:Q", title="매출 금액"),
            tooltip=[alt.Tooltip("항목:N", title="시간대"), alt.Tooltip("값:Q", title="매출 금액", format=",")]
        )
    )
    st.altair_chart(line, use_container_width=True)

# ──────────────────────────────────────────────────────────
# 요일 × 시간대 히트맵 (실제 결합 열 있으면 사용, 없으면 근사)
# ──────────────────────────────────────────────────────────
if weekday_cols_amt and time_cols_amt:
    st.subheader("🗓️⏰ 요일 × 시간대 히트맵")

    def clean_band(c):
        return c.replace("시간대", "").replace("_매출_금액", "").strip("_")

    # 시간대 라벨 후보 추출
    time_bands = sorted(
        { clean_band(c) for c in time_cols_amt },
        key=lambda s: int(re.search(r"(\d+)", s).group(1)) if re.search(r"(\d+)", s) else 999
    )

    # 실제 결합 열 탐지
    real_cells = []
    for w in weekday_order:
        for b in time_bands:
            candidates = [
                f"{w}_시간대_{b}_매출_금액",
                f"시간대_{b}_{w}_매출_금액"
            ]
            col_found = next((cand for cand in candidates if cand in df.columns), None)
            if col_found:
                val = to_num(df[col_found]).sum()
                real_cells.append((w, b, val))

    if real_cells:
        heat_df = pd.DataFrame(real_cells, columns=["요일","시간대","매출"])
        mode_desc = "실제 결합 컬럼 기반"
    else:
        use_approx = st.toggle("실제 결합 열이 없으면 '독립 가정 근사'로 히트맵 생성", value=True)
        if not use_approx:
            st.info("결합 컬럼이 없어 히트맵을 생략합니다. (토글을 켜면 근사로 생성)")
            heat_df = pd.DataFrame(columns=["요일","시간대","매출"])
        else:
            # 요일 합계, 시간대 합계
            wk_tot = sum_by_cols(df, weekday_cols_amt, lambda x: x.replace("_매출_금액",""))
            wk_tot["sort"] = wk_tot["항목"].map({w:i for i,w in enumerate(weekday_order)}).fillna(99)
            wk_tot = wk_tot.sort_values("sort").drop(columns="sort")

            tz_tot = sum_by_cols(df, time_cols_amt, clean_band)
            tz_tot = tz_tot.sort_values(by="항목", key=lambda s: s.map(lambda x: int(re.search(r"(\d+)", x).group(1)) if re.search(r"(\d+)", x) else 999))

            wk_sum = wk_tot["값"].sum()
            tz_sum = tz_tot["값"].sum()
            grand = to_num(df["당월_매출_금액"]).sum() if "당월_매출_금액" in df.columns else min(wk_sum, tz_sum)

            if wk_sum == 0 or tz_sum == 0 or grand == 0:
                st.info("근사 계산에 필요한 합계가 0입니다. 히트맵을 생략합니다.")
                heat_df = pd.DataFrame(columns=["요일","시간대","매출"])
            else:
                wk_tot["w"] = wk_tot["값"] / wk_sum
                tz_tot["w"] = tz_tot["값"] / tz_sum
                heat_df = (
                    wk_tot.assign(key=1)[["항목","w","key"]]
                    .merge(tz_tot.assign(key=1)[["항목","w","key"]], on="key", suffixes=("_요일","_시간대"))
                    .drop(columns="key")
                )
                heat_df.rename(columns={"항목_요일":"요일","항목_시간대":"시간대"}, inplace=True)
                heat_df["매출"] = grand * heat_df["w_요일"] * heat_df["w_시간대"]
                heat_df = heat_df[["요일","시간대","매출"]]
            mode_desc = "독립 가정 근사"

    if not heat_df.empty:
        st.caption(f"히트맵 모드: **{mode_desc}**")
        heat = (
            alt.Chart(heat_df)
            .mark_rect()
            .encode(
                x=alt.X("시간대:N", title="시간대", sort=time_bands),
                y=alt.Y("요일:N", title="요일", sort=weekday_order),
                color=alt.Color("매출:Q", title="매출 금액", scale=alt.Scale(type="linear")),
                tooltip=[alt.Tooltip("요일:N"), alt.Tooltip("시간대:N"), alt.Tooltip("매출:Q", format=",")]
            )
        )
        st.altair_chart(heat, use_container_width=True)
    else:
        st.info("히트맵을 생성할 수 있는 데이터가 없습니다.")

# ──────────────────────────────────────────────────────────
# 성별/연령대
# ──────────────────────────────────────────────────────────
gender_cols_amt = [c for c in df.columns if c.endswith("_매출_금액") and any(g in c for g in ["남성","여성"])]
if gender_cols_amt:
    st.subheader("🚻 성별별 매출 금액")
    def g_clean(x): return x.replace("_매출_금액", "")
    gdf = sum_by_cols(df, gender_cols_amt, g_clean)
    pie = (
        alt.Chart(gdf)
        .mark_arc()
        .encode(
            theta=alt.Theta("값:Q", stack=True),
            color=alt.Color("항목:N", legend=alt.Legend(title="성별")),
            tooltip=[alt.Tooltip("항목:N", title="성별"), alt.Tooltip("값:Q", title="매출 금액", format=",")]
        )
    )
    st.altair_chart(pie, use_container_width=True)

age_tokens = ["10대","20대","30대","40대","50대","60대","60대이상","70대"]
age_cols_amt = [c for c in df.columns if c.endswith("_매출_금액") and any(a in c for a in age_tokens)]
if age_cols_amt:
    st.subheader("🧑‍🧓 연령대별 매출 금액")
    def age_clean(x): return x.replace("_매출_금액", "")
    adf = sum_by_cols(df, age_cols_amt, age_clean)
    order_map = {k:i for i,k in enumerate(["10대","20대","30대","40대","50대","60대","60대이상","70대"])}
    adf["sort"] = adf["항목"].map(lambda s: min([order_map.get(tok, 999) for tok in order_map if tok in s] + [999]))
    adf = adf.sort_values("sort").drop(columns="sort")
    bar = (
        alt.Chart(adf)
        .mark_bar()
        .encode(
            x=alt.X("항목:N", sort=list(adf["항목"]), title="연령대"),
            y=alt.Y("값:Q", title="매출 금액"),
            tooltip=[alt.Tooltip("항목:N", title="연령대"), alt.Tooltip("값:Q", title="매출 금액", format=",")]
        )
    )
    st.altair_chart(bar, use_container_width=True)

# ──────────────────────────────────────────────────────────
# 랭킹 Top-N (업종/상권)
# ──────────────────────────────────────────────────────────
st.subheader(f"🏆 {metric_col} 기준 Top {topn}")
if "서비스_업종_코드_명" in df.columns and metric_col in df.columns:
    top_ind = (
        df.groupby("서비스_업종_코드_명")[metric_col]
        .apply(lambda s: to_num(s).sum())
        .reset_index()
        .sort_values(by=metric_col, ascending=False)
        .head(topn)
    )
    c1, c2 = st.columns(2)
    with c1:
        st.caption("업종 Top-N 표")
        st.dataframe(top_ind, use_container_width=True)
    with c2:
        chart = (
            alt.Chart(top_ind)
            .mark_bar()
            .encode(
                x=alt.X(f"{metric_col}:Q", title=metric_col, sort="descending"),
                y=alt.Y("서비스_업종_코드_명:N", sort="-x", title="업종"),
                tooltip=["서비스_업종_코드_명", alt.Tooltip(metric_col, format=",")]
            )
        )
        st.altair_chart(chart, use_container_width=True)

if "상권_코드_명" in df.columns and metric_col in df.columns:
    top_area = (
        df.groupby("상권_코드_명")[metric_col]
        .apply(lambda s: to_num(s).sum())
        .reset_index()
        .sort_values(by=metric_col, ascending=False)
        .head(topn)
    )
    c3, c4 = st.columns(2)
    with c3:
        st.caption("상권 Top-N 표")
        st.dataframe(top_area, use_container_width=True)
    with c4:
        chart = (
            alt.Chart(top_area)
            .mark_bar()
            .encode(
                x=alt.X(f"{metric_col}:Q", title=metric_col, sort="descending"),
                y=alt.Y("상권_코드_명:N", sort="-x", title="상권"),
                tooltip=["상권_코드_명", alt.Tooltip(metric_col, format=",")]
            )
        )
        st.altair_chart(chart, use_container_width=True)

# ──────────────────────────────────────────────────────────
# 테이블 + 다운로드
# ──────────────────────────────────────────────────────────
st.subheader("📥 현재 필터 반영 데이터")
st.dataframe(df, use_container_width=True)
csv = df.to_csv(index=False, encoding="utf-8-sig")
st.download_button("CSV 다운로드", data=csv, file_name="filtered_data.csv", mime="text/csv")
