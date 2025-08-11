# main.py
import os, re
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="서울시 상권 분기 매출 분석 대시보드", layout="wide")

# ─────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(file_like_or_path):
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(file_like_or_path, encoding=enc)
        except Exception:
            continue
    st.error("CSV를 읽지 못했습니다. 인코딩을 확인하세요.")
    return None

def to_num(s):
    return pd.to_numeric(pd.Series(s, dtype="object").astype(str).str.replace(",", "", regex=False), errors="coerce")

def find_col(df, regex_list):
    cols = list(df.columns)
    for rx in regex_list:
        pat = re.compile(rx)
        for c in cols:
            if pat.search(str(c)):
                return c
    return None

def parse_yq(x):
    s = str(x)
    m = re.search(r"(20\d{2}).*?([1-4])", s)
    if m: return int(m.group(1)), int(m.group(2))
    if s.isdigit() and len(s) in (5,6): return int(s[:4]), int(s[-1])
    return (9999,9)

def qprev(y,q):  # 직전분기
    return (y-1,4) if q==1 else (y, q-1)

def make_qkey_series(qseries):
    return qseries.astype(str).map(parse_yq)

def pct(a,b):
    return (a/b-1)*100 if (b is not None and b!=0) else np.nan

def safe_sum(df, col):
    return to_num(df[col]).sum()

def rank_df(df, bycol, topn=10, ascending=False):
    return df.sort_values(by=bycol, ascending=ascending).head(topn)

# ─────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────
st.title("📊 서울시 상권 **분기** 매출 분석 대시보드 (원본 포맷 그대로)")

up = st.file_uploader("CSV 업로드 (없으면 샘플 자동 로드)", type=["csv"])
if up:
    df = load_csv(up)
else:
    st.info("샘플 데이터를 로드했습니다. (같은 경로의 CSV)")
    sample = os.path.join(os.path.dirname(__file__), "서울시 상권분석서비스(추정매출-상권)_sample.csv")
    if os.path.exists(sample):
        df = load_csv(sample)
    else:
        st.error("샘플 CSV가 없습니다."); st.stop()

if df is None or df.empty:
    st.error("데이터가 비었습니다."); st.stop()

# 컬럼 정리(앞뒤 공백/숨은문자 제거)
df.columns = df.columns.map(lambda c: str(c).strip().replace("\u200b",""))

# 핵심 컬럼 자동탐지 (데이터셋 명칭 차이를 흡수)
COL_Q   = find_col(df, [r"기준.?년분기.?코드", r"^분기$", r"년.?분기"])
COL_AMT = find_col(df, [r"분기.?매출.?금액", r"매출.?금액"])
COL_CNT = find_col(df, [r"분기.?매출.?건수", r"매출.?건수"])
COL_AREA   = find_col(df, [r"상권.?코드.?명"])
COL_AREA_CD= find_col(df, [r"상권.?코드(?!.*명)"])
COL_UPJONG = find_col(df, [r"서비스.?업종.?코드.?명"])
COL_GU     = find_col(df, [r"자치구|구청|구.?명"])

for need, nm in [(COL_Q,"분기 코드"),(COL_AMT,"분기 매출 금액"),(COL_CNT,"분기 매출 건수")]:
    if need is None:
        st.error(f"핵심 컬럼을 찾지 못했습니다: {nm}")
        st.stop()

# 분기 키/정렬키
df["__yq__"] = make_qkey_series(df[COL_Q])
df = df[~df["__yq__"].isna()]  # 혹시 이상치 제거
df["__yq_sort__"] = df["__yq__"].map(lambda t: t[0]*10+t[1])

# ─────────────────────────────────────────
# 사이드바 (공통 필터)
# ─────────────────────────────────────────
with st.sidebar:
    st.header("🔎 필터")
    # 기간 필터(최근 N개 분기)
    uniq_q = sorted(df["__yq__"].unique(), key=lambda t:(t[0],t[1]))
    default_last = uniq_q[-5:] if len(uniq_q)>=5 else uniq_q
    picked_q = st.multiselect(
        "분기 선택",
        options=[f"{y}Q{q}" for (y,q) in uniq_q],
        default=[f"{y}Q{q}" for (y,q) in default_last]
    )
    picked_tuple = set(tuple(map(int, re.findall(r"\d+", s))) for s in picked_q) if picked_q else set(uniq_q)
    # 상권/업종/구 필터
    chosen = {}
    for label, col in [("상권",COL_AREA),("업종",COL_UPJONG),("자치구",COL_GU)]:
        if col and col in df.columns:
            vals = sorted(df[col].dropna().astype(str).unique())
            sel = st.multiselect(f"{label} 선택", vals)
            if sel: chosen[col]=sel
    st.divider()
    metric_pick = st.radio("지표 선택", ["매출 금액","매출 건수"], index=0)
    TOPN = st.slider("Top-N", 5, 50, 20, 1)

# 필터 적용
work = df[df["__yq__"].isin(picked_tuple)].copy()
for c,vals in chosen.items():
    work = work[work[c].astype(str).isin(vals)]
if work.empty:
    st.warning("필터 결과가 없습니다. 조건을 완화하세요."); st.stop()

# 분석에 쓸 실제 지표 컬럼
METRIC = COL_AMT if metric_pick=="매출 금액" else COL_CNT

# ─────────────────────────────────────────
# 레이아웃 탭
# ─────────────────────────────────────────
tab_overview, tab_trend, tab_mix, tab_rank, tab_compare, tab_data = st.tabs(
    ["📌 개요", "📈 추이", "🧩 믹스", "🏆 랭킹·급등락", "🆚 비교(A/B/도시평균)", "📥 데이터/내보내기"]
)

# ========== 개요 ==========
with tab_overview:
    st.subheader("요약 KPI (선택 기간 합계)")
    total_amt = safe_sum(work, COL_AMT)
    total_cnt = safe_sum(work, COL_CNT)
    avg_price = (total_amt/total_cnt) if total_cnt else np.nan

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("총 매출 금액", f"{total_amt:,.0f} 원")
    c2.metric("총 매출 건수", f"{total_cnt:,.0f} 건")
    c3.metric("평균 객단가", f"{avg_price:,.0f} 원" if pd.notna(avg_price) else "계산 불가")

    # 최근분기/직전분기/전년동분기
    st.divider()
    last_yq = sorted(work["__yq__"].unique(), key=lambda t:(t[0],t[1]))[-1]
    y,q = last_yq
    py,pq = qprev(y,q)
    ly = y-1

    def sum_q(y,q,col):
        sub = work[(work["__yq__"]==(y,q))]
        return to_num(sub[col]).sum()

    cur = sum_q(y,q,COL_AMT); prev = sum_q(py,pq,COL_AMT); yoy = sum_q(ly,q,COL_AMT)
    c5,c6,c7 = st.columns(3)
    c5.metric(f"{y}Q{q} 매출", f"{cur:,.0f} 원")
    c6.metric("QoQ 변화", f"{pct(cur,prev):.1f} %" if prev else "N/A")
    c7.metric("YoY 변화", f"{pct(cur,yoy):.1f} %" if yoy else "N/A")

    # 업종 기여도(워터폴 모사 = 기준선 + 업종별 증감 막대)
    if COL_UPJONG in work.columns:
        st.subheader("전분기 대비 업종별 매출 증감 기여도 (선택 상권/필터 반영)")
        cur_up = work[work["__yq__"]==(y,q)].groupby(COL_UPJONG)[COL_AMT].apply(to_num).sum()
        prev_up= work[work["__yq__"]==(py,pq)].groupby(COL_UPJONG)[COL_AMT].apply(to_num).sum()
        diff = (cur_up - prev_up).rename("증감")
        wf = diff.reset_index().sort_values("증감", ascending=False)
        base = 0; seg=[]
        for _,row in wf.iterrows():
            start=base; end=base+row["증감"]; seg.append([row[COL_UPJONG], start, end]); base=end
        if not wf.empty:
            wf_plot = pd.DataFrame(seg, columns=["업종","start","end"])
            wf_plot["증감"] = wf_plot["end"]-wf_plot["start"]
            st.altair_chart(
                alt.Chart(wf_plot).mark_bar().encode(
                    x=alt.X("업종:N", sort="-y"),
                    y=alt.Y("end:Q"),
                    y2="start:Q",
                    color=alt.Color("증감:Q", scale=alt.Scale(scheme="redblue")),
                    tooltip=["업종","증감"]
                ),
                use_container_width=True
            )
        else:
            st.info("업종 컬럼 없음 또는 데이터 부족")

# ========== 추이 ==========
with tab_trend:
    st.subheader("분기별 추이 (총액)")
    trend = work.groupby("__yq__")[COL_AMT].apply(to_num).sum().reset_index()
    trend["분기"] = trend["__yq__"].map(lambda t: f"{t[0]}Q{t[1]}")
    st.altair_chart(
        alt.Chart(trend.sort_values("__yq__")).mark_line(point=True).encode(
            x="분기:N", y=alt.Y(f"{COL_AMT}:Q", title="매출 금액"),
            tooltip=["분기", alt.Tooltip(COL_AMT, format=",")]
        ),
        use_container_width=True
    )

    # 업종/상권 드릴다운
    c1,c2 = st.columns(2)
    if COL_UPJONG in work.columns:
        with c1:
            st.caption("업종별 분기 추이 (Top-N by 최근분기)")
            last_slice = work[work["__yq__"]==last_yq]
            top_up = last_slice.groupby(COL_UPJONG)[COL_AMT].apply(to_num).sum().sort_values(ascending=False).head(10).index
            sub = work[work[COL_UPJONG].isin(top_up)].copy()
            sub["분기"] = sub["__yq__"].map(lambda t: f"{t[0]}Q{t[1]}")
            agg = sub.groupby([COL_UPJONG,"분기"])[COL_AMT].apply(to_num).sum().reset_index()
            st.altair_chart(
                alt.Chart(agg).mark_line(point=True).encode(
                    x="분기:N", y=alt.Y(f"{COL_AMT}:Q", title="매출 금액"),
                    color=alt.Color(f"{COL_UPJONG}:N", legend=alt.Legend(title="업종")),
                    tooltip=[COL_UPJONG,"분기", alt.Tooltip(COL_AMT, format=",")]
                ),
                use_container_width=True
            )
    if COL_AREA in work.columns:
        with c2:
            st.caption("상권별 분기 추이 (Top-N by 최근분기)")
            last_slice = work[work["__yq__"]==last_yq]
            top_ar = last_slice.groupby(COL_AREA)[COL_AMT].apply(to_num).sum().sort_values(ascending=False).head(10).index
            sub = work[work[COL_AREA].isin(top_ar)].copy()
            sub["분기"] = sub["__yq__"].map(lambda t: f"{t[0]}Q{t[1]}")
            agg = sub.groupby([COL_AREA,"분기"])[COL_AMT].apply(to_num).sum().reset_index()
            st.altair_chart(
                alt.Chart(agg).mark_line(point=True).encode(
                    x="분기:N", y=alt.Y(f"{COL_AMT}:Q", title="매출 금액"),
                    color=alt.Color(f"{COL_AREA}:N", legend=alt.Legend(title="상권")),
                    tooltip=[COL_AREA,"분기", alt.Tooltip(COL_AMT, format=",")]
                ),
                use_container_width=True
            )

# ========== 믹스 ==========
with tab_mix:
    st.subheader("소비 패턴 믹스")
    # 요일/시간/성별/연령대 자동 탐지
    weekday_order = ["월요일","화요일","수요일","목요일","금요일","토요일","일요일"]
    w_cols = [c for c in work.columns if c.endswith("_매출_금액") and any(w in c for w in weekday_order)]
    t_cols = [c for c in work.columns if ("시간대" in c and c.endswith("_매출_금액"))]
    g_cols = [c for c in work.columns if c.endswith("_매출_금액") and any(g in c for g in ["남성","여성"])]
    a_tokens = ["10대","20대","30대","40대","50대","60대","60대이상","70대"]
    a_cols = [c for c in work.columns if c.endswith("_매출_금액") and any(a in c for a in a_tokens)]

    def sum_by_cols(cols, cleaner=lambda x:x):
        if not cols: return pd.DataFrame()
        s = work[cols].apply(to_num).sum(axis=0)
        out = pd.DataFrame({"항목": s.index, "값": s.values})
        out["항목"] = out["항목"].map(cleaner)
        return out

    # 요일 막대
    if w_cols:
        def wk_clean(x): return x.replace("_매출_금액","")
        wk = sum_by_cols(w_cols, wk_clean)
        wk["sort"] = wk["항목"].map({w:i for i,w in enumerate(weekday_order)}); wk=wk.sort_values("sort")
        st.altair_chart(
            alt.Chart(wk).mark_bar().encode(
                x=alt.X("항목:N", title="요일"), y=alt.Y("값:Q", title="매출 금액"),
                tooltip=[alt.Tooltip("항목", title="요일"), alt.Tooltip("값:Q", format=",")]
            ),
            use_container_width=True
        )

    # 시간대 라인
    if t_cols:
        def tz_clean(x): return x.replace("시간대","").replace("_매출_금액","").strip("_")
        tz = sum_by_cols(t_cols, tz_clean)
        def tz_key(s):
            m=re.search(r"(\d+)", str(s)); return int(m.group(1)) if m else 999
        tz = tz.sort_values(by="항목", key=lambda s:s.map(tz_key))
        st.altair_chart(
            alt.Chart(tz).mark_line(point=True).encode(
                x=alt.X("항목:N", title="시간대"), y=alt.Y("값:Q", title="매출 금액"),
                tooltip=["항목", alt.Tooltip("값:Q", format=",")]
            ),
            use_container_width=True
        )

    # 히트맵(요일×시간대)
    if w_cols and t_cols:
        st.caption("요일×시간대 히트맵")
        # 결합 열 우선 탐지
        def band_clean(c): return c.replace("시간대","").replace("_매출_금액","").strip("_")
        bands = sorted({band_clean(c) for c in t_cols}, key=lambda s:int(re.search(r"(\d+)",s).group(1)))
        real=[]
        for w in weekday_order:
            for b in bands:
                for cand in (f"{w}_시간대_{b}_매출_금액", f"시간대_{b}_{w}_매출_금액"):
                    if cand in work.columns:
                        real.append((w,b,to_num(work[cand]).sum())); break
        if real:
            heat = pd.DataFrame(real, columns=["요일","시간대","매출"])
        else:
            # 독립 가정 근사
            wk = sum_by_cols(w_cols, wk_clean); tz = sum_by_cols(t_cols, band_clean)
            wk_w = wk["값"]/wk["값"].sum() if wk["값"].sum()!=0 else 0
            tz_w = tz["값"]/tz["값"].sum() if tz["값"].sum()!=0 else 0
            base = safe_sum(work, COL_AMT)
            heat = pd.DataFrame([(w,b, base*wk_w.iloc[i]*tz_w.iloc[j])
                                 for i,w in enumerate(wk["항목"])
                                 for j,b in enumerate(tz["항목"])], columns=["요일","시간대","매출"])
        st.altair_chart(
            alt.Chart(heat).mark_rect().encode(
                x=alt.X("시간대:N", sort=bands), y=alt.Y("요일:N", sort=weekday_order),
                color=alt.Color("매출:Q", title="매출 금액"),
                tooltip=["요일","시간대", alt.Tooltip("매출:Q", format=",")]
            ),
            use_container_width=True
        )

    # 성별/연령대
    if g_cols:
        g = sum_by_cols(g_cols, lambda x: x.replace("_매출_금액",""))
        st.altair_chart(
            alt.Chart(g).mark_arc().encode(
                theta="값:Q", color=alt.Color("항목:N", legend=alt.Legend(title="성별")),
                tooltip=["항목", alt.Tooltip("값:Q", format=",")]
            ),
            use_container_width=True
        )
    if a_cols:
        a = sum_by_cols(a_cols, lambda x: x.replace("_매출_금액",""))
        order = {k:i for i,k in enumerate(a_tokens)}
        a["sort"]=a["항목"].map(lambda s: min([order.get(t,999) for t in a_tokens if t in s]+[999]))
        a=a.sort_values("sort")
        st.altair_chart(
            alt.Chart(a).mark_bar().encode(
                x=alt.X("항목:N", title="연령대"), y=alt.Y("값:Q", title="매출 금액"),
                tooltip=["항목", alt.Tooltip("값:Q", format=",")]
            ),
            use_container_width=True
        )

# ========== 랭킹/급등락 ==========
with tab_rank:
    st.subheader("랭킹 & 급등락 분석")
    base_col = COL_UPJONG if COL_UPJONG in work.columns else COL_AREA if COL_AREA in work.columns else None
    if base_col is None:
        st.info("랭킹을 계산할 그룹 컬럼(업종/상권)이 없습니다.")
    else:
        # 최근분기 기준 랭킹
        last_slice = work[work["__yq__"]==last_yq]
        cur_rank = last_slice.groupby(base_col)[METRIC].apply(to_num).sum().reset_index().rename(columns={METRIC:"값"})
        top_now = rank_df(cur_rank, "값", TOPN, ascending=False)
        c1,c2 = st.columns(2)
        with c1:
            st.caption(f"최근분기 Top {TOPN} ({base_col})")
            st.dataframe(top_now, use_container_width=True)
        with c2:
            st.altair_chart(
                alt.Chart(top_now).mark_bar().encode(
                    x=alt.X("값:Q", title=metric_pick, sort="descending"),
                    y=alt.Y(f"{base_col}:N", sort="-x"),
                    tooltip=[base_col, alt.Tooltip("값:Q", format=",")]
                ),
                use_container_width=True
            )

        # QoQ/YoY 급등락
        st.caption("QoQ / YoY 변화율 (최근분기 기준)")
        prev_slice = work[work["__yq__"]==qprev(*last_yq)]
        yoy_slice  = work[work["__yq__"]==(last_yq[0]-1, last_yq[1])]
        comp = cur_rank.merge(
            prev_slice.groupby(base_col)[METRIC].apply(to_num).sum().reset_index().rename(columns={METRIC:"직전"}),
            on=base_col, how="left"
        ).merge(
            yoy_slice.groupby(base_col)[METRIC].apply(to_num).sum().reset_index().rename(columns={METRIC:"전년동분기"}),
            on=base_col, how="left"
        )
        comp["QoQ(%)"] = comp.apply(lambda r: pct(r["값"], r["직전"]), axis=1)
        comp["YoY(%)"] = comp.apply(lambda r: pct(r["값"], r["전년동분기"]), axis=1)
        # 간단 이상치 감지(Z-score)
        comp["z_QoQ"] = ((comp["QoQ(%)"] - comp["QoQ(%)"].mean())/comp["QoQ(%)"].std(ddof=0)).replace([np.inf,-np.inf], np.nan)
        movers = comp.sort_values("QoQ(%)", ascending=False).head(TOPN)
        st.dataframe(movers[[base_col,"값","직전","전년동분기","QoQ(%)","YoY(%)","z_QoQ"]], use_container_width=True)

# ========== 비교 ==========
with tab_compare:
    st.subheader("A/B/도시평균 비교")
    # 비교 축 선택(업종/상권/자치구 중 존재하는 것)
    comp_cols = [(label,col) for label,col in [("상권",COL_AREA),("업종",COL_UPJONG),("자치구",COL_GU)] if col]
    if not comp_cols:
        st.info("비교 가능한 분류 컬럼이 없습니다.")
    else:
        labelA,colA = comp_cols[0]
        opts = sorted(work[colA].dropna().astype(str).unique())
        A = st.selectbox(f"A {labelA} 선택", opts, index=0)
        B = st.selectbox(f"B {labelA} 선택", opts, index=1 if len(opts)>1 else 0)

        def series_by_yq(df_, keyval):
            sub = df_[df_[colA].astype(str)==str(keyval)]
            return sub.groupby("__yq__")[METRIC].apply(to_num).sum()

        sA = series_by_yq(work, A); sB = series_by_yq(work, B)
        city = work.groupby("__yq__")[METRIC].apply(to_num).sum()

        comp_df = pd.DataFrame({
            "A": sA, "B": sB, "City": city
        }).reset_index().rename(columns={"__yq__":"yq"})
        comp_df["분기"] = comp_df["yq"].map(lambda t: f"{t[0]}Q{t[1]}")
        # Index=100 (도시평균 대비)
        comp_df["A_idx"] = (comp_df["A"]/comp_df["City"])*100
        comp_df["B_idx"] = (comp_df["B"]/comp_df["City"])*100

        c1,c2 = st.columns(2)
        with c1:
            st.caption("절대값 추이")
            st.altair_chart(
                alt.Chart(comp_df).transform_fold(
                    ["A","B","City"], as_=["분류","값"]
                ).mark_line(point=True).encode(
                    x="분기:N", y=alt.Y("값:Q", title=metric_pick),
                    color=alt.Color("분류:N"),
                    tooltip=["분기","분류", alt.Tooltip("값:Q", format=",")]
                ),
                use_container_width=True
            )
        with c2:
            st.caption("도시평균=100 지수 비교")
            st.altair_chart(
                alt.Chart(comp_df).transform_fold(
                    ["A_idx","B_idx"], as_=["분류","지수"]
                ).mark_line(point=True).encode(
                    x="분기:N", y=alt.Y("지수:Q", title="Index(도시=100)"),
                    color=alt.Color("분류:N", legend=alt.Legend(title=f"{labelA} 지수")),
                    tooltip=["분기","분류", alt.Tooltip("지수:Q", format=".1f")]
                ),
                use_container_width=True
            )

# ========== 데이터 ==========
with tab_data:
    st.subheader("현재 필터 반영 데이터")
    st.dataframe(work.drop(columns=["__yq__","__yq_sort__"]), use_container_width=True)
    st.download_button(
        "CSV 다운로드",
        data=work.drop(columns=["__yq__","__yq_sort__"]).to_csv(index=False, encoding="utf-8-sig"),
        file_name="filtered_data.csv",
        mime="text/csv"
    )
