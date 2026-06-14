"""System B Opportunity Lab dashboard.

Run with:
    streamlit run dashboards/system_b_demo/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
SYSTEM_B_DIR = PROJECT_ROOT / "data" / "processed" / "system_b"

try:
    import plotly.express as px
    import streamlit as st
except ImportError:
    print("Install dashboard dependencies: pip install streamlit plotly")
    sys.exit(1)


st.set_page_config(
    page_title="System B - Opportunity Lab",
    page_icon="NIP",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background: #f8fafc; color: #111827; }
    .block-container { max-width: 1240px; padding-top: 2rem; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
    h1, h2, h3, p, li, label, .stMarkdown { color: #111827; }
    .hero {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 20px 22px;
        margin-bottom: 18px;
    }
    .note {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        padding: 14px 16px;
        color: #7c2d12;
        margin-top: 16px;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
    }
    .term {
        border-bottom: 1px dotted #2563eb;
        color: #1d4ed8;
        cursor: help;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_parquet(name: str) -> pd.DataFrame:
    path = SYSTEM_B_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


promotion = load_parquet("promotion_scores.parquet")
bandit = load_parquet("bandit_policy_metrics.parquet")
fairness = load_parquet("fairness_metrics.parquet")
frontier = load_parquet("pareto_frontier.parquet")
ips = load_parquet("ips_stress_test.parquet")
exposure = load_parquet("exposure_log.parquet")

with st.sidebar:
    st.title("System B")
    section = st.radio(
        "View",
        ["Overview", "Opportunity Scout", "Bandit Policies", "Fairness", "IPS Stress Test"],
    )
    st.caption("Content & Creator Opportunity Intelligence Lab")

st.markdown(
    """
<div class="hero">
  <h1>System B - Content & Creator Opportunity Lab</h1>
  <p>Find underexposed content with measurable upside, compare exploration policies, and audit exposure concentration before simulated deployment.</p>
</div>
""",
    unsafe_allow_html=True,
)

if promotion.empty:
    st.error("System B artifacts not found. Run: python scripts/run_system_b_pipeline.py")
    st.stop()

if section == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Items Scored", f"{len(promotion):,}")
    c2.metric("Exposure Rows", f"{len(exposure):,}" if not exposure.empty else "0")
    c3.metric("Creators", f"{promotion['creator_id'].nunique():,}" if "creator_id" in promotion else "0")
    c4.metric("Top Promotion", f"{promotion['promotion_score'].max():.3f}")

    st.subheader("Top Opportunity Items")
    cols = [
        "item_id",
        "title",
        "creator_id",
        "primary_genre",
        "promotion_score",
        "shrunk_mean",
        "breakout_score",
        "uplift_score",
        "posterior_uncertainty",
    ]
    st.dataframe(
        promotion.sort_values("promotion_score", ascending=False)[[c for c in cols if c in promotion.columns]].head(20),
        use_container_width=True,
    )

    st.markdown(
        """
<div class="note">
This dashboard uses simulated exposure logs with known propensities. It demonstrates policy design, uncertainty, and offline-evaluation mechanics; it is not live production traffic.
</div>
""",
        unsafe_allow_html=True,
    )

elif section == "Opportunity Scout":
    genre = st.selectbox("Genre", ["All"] + sorted(promotion.get("primary_genre", pd.Series(["Unknown"])).dropna().astype(str).unique().tolist()))
    view = promotion.copy()
    if genre != "All" and "primary_genre" in view.columns:
        view = view[view["primary_genre"].astype(str).eq(genre)]
    view = view.sort_values("promotion_score", ascending=False).head(50)
    st.plotly_chart(
        px.scatter(
            view,
            x="shrunk_mean",
            y="uplift_score",
            size="posterior_uncertainty",
            color="primary_genre" if "primary_genre" in view.columns else None,
            hover_data=["item_id", "title", "creator_id"],
            title="Quality vs Uplift, Sized by Uncertainty",
        ),
        use_container_width=True,
    )
    st.dataframe(view, use_container_width=True)

elif section == "Bandit Policies":
    if bandit.empty:
        st.warning("No bandit policy metrics found.")
    else:
        st.plotly_chart(
            px.line(bandit, x="round", y="cumulative_regret", color="policy", title="Cumulative Regret"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.line(bandit, x="round", y="unique_items_exposed", color="policy", title="Exploration Breadth"),
            use_container_width=True,
        )

elif section == "Fairness":
    if fairness.empty or frontier.empty:
        st.warning("No fairness artifacts found.")
    else:
        st.plotly_chart(
            px.line(fairness, x="day", y="gini", color="policy", title="Creator Exposure Gini Over Time"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.scatter(
                frontier,
                x="gini",
                y="mean_relevance",
                color="mean_novelty",
                hover_data=["lambda_novelty", "lambda_fairness", "long_tail_viability"],
                title="Pareto Frontier: Relevance vs Exposure Concentration",
            ),
            use_container_width=True,
        )
        st.dataframe(frontier.head(20), use_container_width=True)

elif section == "IPS Stress Test":
    if ips.empty:
        st.warning("No IPS stress-test artifact found.")
    else:
        st.plotly_chart(
            px.bar(ips, x="target_policy", y=["ips", "snips", "clipped_ips_10", "doubly_robust"], barmode="group", title="Off-Policy Value Estimates"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.bar(ips, x="target_policy", y="effective_sample_size", title="Effective Sample Size by Policy Distance"),
            use_container_width=True,
        )
        st.dataframe(ips, use_container_width=True)
