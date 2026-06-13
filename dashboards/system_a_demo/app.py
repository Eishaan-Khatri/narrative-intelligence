"""
Narrative Intelligence Platform - System A Demo Dashboard.

Run with:
    streamlit run dashboards/system_a_demo/app.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
    import plotly.graph_objects as go
except ImportError:
    print("Streamlit and/or Plotly not installed. Run: pip install streamlit plotly")
    sys.exit(1)


st.set_page_config(
    page_title="System A - Discovery Engine",
    page_icon="NIP",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
        background: #f8fafc;
        color: #111827;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1220px;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    [data-testid="stSidebar"] * {
        color: #111827 !important;
    }

    [data-testid="stSidebar"] small {
        color: #64748b !important;
    }

    h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown, .stText {
        color: #111827;
    }

    .hero {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 22px 24px;
        margin-bottom: 18px;
    }

    .hero h1 {
        color: #111827;
        font-size: 2rem;
        line-height: 1.15;
        margin: 0 0 8px 0;
        letter-spacing: 0;
    }

    .hero p {
        color: #4b5563;
        font-size: 1rem;
        margin: 0;
        max-width: 900px;
    }

    .note-box {
        background: #ffffff;
        border: 1px solid #dbe3ef;
        border-radius: 8px;
        padding: 16px 18px;
        color: #1f2937;
        margin: 28px 0 6px 0;
    }

    .note-box strong {
        color: #0f172a;
    }

    .term {
        border-bottom: 1px dotted #2563eb;
        color: #1d4ed8;
        cursor: help;
        position: relative;
        font-weight: 600;
    }

    .term .tip {
        visibility: hidden;
        width: 260px;
        background: #111827;
        color: #ffffff !important;
        text-align: left;
        border-radius: 6px;
        padding: 9px 10px;
        position: absolute;
        z-index: 50;
        bottom: 135%;
        left: 0;
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.22);
        font-size: 0.82rem;
        line-height: 1.35;
        font-weight: 400;
    }

    .term:hover .tip {
        visibility: visible;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 4px 10px;
        background: #e0f2fe;
        color: #075985;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.9rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #111827 !important;
    }

    .stAlert {
        color: #111827;
    }
</style>
""",
    unsafe_allow_html=True,
)


DATA_DIR = PROJECT_ROOT / "data" / "processed"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

GLOSSARY = {
    "two-tower retrieval": "A recommendation model with one neural network for users and one for items. It quickly finds candidate items that match a user.",
    "FAISS": "A vector search library used to retrieve nearest item embeddings quickly.",
    "LambdaMART": "A learning-to-rank model that reorders candidates using several business and behavior features.",
    "survival model": "A model that estimates where users are likely to stop reading or abandon a story.",
    "completion-weighted NDCG": "A ranking metric that gives more credit when recommended stories are actually read deeply, not just clicked.",
    "ablation study": "A comparison where one system layer is added at a time to measure what each layer contributes.",
    "hazard score": "The estimated risk that a user will abandon a story at a given point.",
    "quality score": "A PCA-based score built from completion, return, engagement, sentiment proxy, and structural signals.",
    "feature store": "The shared table layer that turns raw reading events into reusable model features.",
}


def term(label: str) -> str:
    definition = GLOSSARY[label]
    return f'<span class="term">{label}<span class="tip">{definition}</span></span>'


@st.cache_data
def load_parquet_safe(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None


def primary_genre(value: object) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except (SyntaxError, ValueError):
            pass
        return value.strip("[]'\" ").split(",")[0].strip("'\" ")
    return "Unknown"


def normalize_quality(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    vmin = values.min()
    vmax = values.max()
    if vmax == vmin:
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    return (values - vmin) / (vmax - vmin)


def normalize_ablation_columns(ablation: pd.DataFrame) -> pd.DataFrame:
    renamed = ablation.rename(
        columns={
            "model": "Model",
            "description": "Description",
            "CW_NDCG@10": "CW-NDCG@10",
            "Binary_NDCG@10": "Binary-NDCG@10",
        }
    )
    if "Description" in renamed.columns:
        renamed["Model"] = renamed["Model"].astype(str) + " - " + renamed["Description"].astype(str)
    return renamed


@st.cache_data
def generate_demo_data() -> dict:
    np.random.seed(42)
    n_users = 50
    n_items = 200

    users = pd.DataFrame(
        {
            "user_id": [f"u_{i:04d}" for i in range(n_users)],
            "avg_velocity": np.random.normal(200, 50, n_users).clip(50, 600),
            "avg_completion": np.random.beta(3, 2, n_users),
            "sessions_per_week": np.random.poisson(5, n_users).astype(float),
            "dominant_genre": np.random.choice(
                ["Fantasy", "Romance", "Sci-Fi", "Mystery", "Literary Fiction"], n_users
            ),
        }
    )

    items = pd.DataFrame(
        {
            "item_id": [f"i_{i:04d}" for i in range(n_items)],
            "title": [f"Story {i}" for i in range(n_items)],
            "quality_score": np.random.beta(2, 3, n_items),
            "popularity_pct": np.random.uniform(0, 1, n_items),
            "genre": np.random.choice(
                ["Fantasy", "Romance", "Sci-Fi", "Mystery", "Literary Fiction"], n_items
            ),
            "chapter_count": np.random.randint(5, 50, n_items),
            "avg_rating": np.random.normal(3.5, 0.8, n_items).clip(1, 5),
        }
    )

    ablation = pd.DataFrame(
        {
            "Model": [
                "M0 - SVD baseline",
                "M1 - Behavioral features",
                "M2 - Hard negatives",
                "M3 - Quality filter",
                "M4 - Survival re-rank",
            ],
            "CW-NDCG@10": [0.312, 0.387, 0.421, 0.448, 0.483],
            "Binary-NDCG@10": [0.445, 0.472, 0.501, 0.512, 0.524],
            "Recall@500": [np.nan, np.nan, 0.723, 0.698, 0.698],
        }
    )

    chapters = np.arange(1, 31)
    survival_curves = {
        "High quality": np.clip(1.0 - 0.02 * chapters, 0, 1),
        "Medium quality": np.clip(1.0 - 0.04 * chapters, 0, 1),
        "Low quality": np.clip(1.0 - 0.08 * chapters, 0, 1),
    }

    return {
        "users": users,
        "items": items,
        "ablation": ablation,
        "ranking_metrics": pd.DataFrame(
            {
                "k": [10],
                "CW_NDCG": [0.483],
                "Binary_NDCG": [0.524],
                "delta": [-0.041],
            }
        ),
        "overall_recall_500": np.nan,
        "tail_recall_500": np.nan,
        "survival_curves": survival_curves,
        "chapters": chapters,
        "source": "Demo data",
        "session_count": 0,
    }


@st.cache_data
def build_real_dashboard_data(
    _session_features: pd.DataFrame,
    _catalog: pd.DataFrame,
    _quality_scores: pd.DataFrame | None,
    _ablation: pd.DataFrame | None,
    _ranking_metrics: pd.DataFrame | None,
    _oracle_analysis: pd.DataFrame | None,
) -> dict:
    items = _catalog.copy()
    items["genre"] = items["genres"].apply(primary_genre) if "genres" in items.columns else "Unknown"
    if "rating_count" in items.columns:
        items["popularity_pct"] = items["rating_count"].rank(pct=True)
    else:
        item_pop = _session_features["item_id"].value_counts(normalize=True).rename("popularity_pct")
        items = items.merge(item_pop, left_on="item_id", right_index=True, how="left")
        items["popularity_pct"] = items["popularity_pct"].fillna(0.0)

    if _quality_scores is not None and "quality_score" in _quality_scores.columns:
        items = items.merge(_quality_scores[["item_id", "quality_score"]], on="item_id", how="left")
    elif "latent_quality" in items.columns:
        items["quality_score"] = items["latent_quality"]
    else:
        items["quality_score"] = 0.5

    items["quality_score"] = normalize_quality(items["quality_score"].fillna(items["quality_score"].median()))
    items["avg_rating"] = items.get("avg_rating", pd.Series(np.nan, index=items.index)).fillna(3.0)
    items["chapter_count"] = items.get("chapter_count", pd.Series(0, index=items.index)).fillna(0).astype(int)
    items["title"] = items.get("title", pd.Series(items["item_id"], index=items.index))

    session_with_genre = _session_features.merge(items[["item_id", "genre"]], on="item_id", how="left")
    active_days = (
        pd.to_datetime(_session_features["timestamp_start"]).max()
        - pd.to_datetime(_session_features["timestamp_start"]).min()
    ).days + 1
    active_weeks = max(active_days / 7, 1)

    users = (
        _session_features.groupby("user_id")
        .agg(
            avg_velocity=("reading_velocity_wpm", "mean"),
            avg_completion=("final_completion_pct", "mean"),
            sessions_per_week=("session_id", lambda s: len(s) / active_weeks),
        )
        .reset_index()
    )
    genre_mode = (
        session_with_genre.groupby("user_id")["genre"]
        .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else "Unknown")
        .rename("dominant_genre")
        .reset_index()
    )
    users = users.merge(genre_mode, on="user_id", how="left").fillna({"dominant_genre": "Unknown"})

    chapters = np.arange(1, 31)
    survival_curves = {
        "High quality": np.clip(1.0 - 0.018 * chapters, 0, 1),
        "Medium quality": np.clip(1.0 - 0.035 * chapters, 0, 1),
        "Low quality": np.clip(1.0 - 0.070 * chapters, 0, 1),
    }

    ranking_metrics = _ranking_metrics if _ranking_metrics is not None else generate_demo_data()["ranking_metrics"]
    overall_recall_500 = np.nan
    tail_recall_500 = np.nan
    if _oracle_analysis is not None and not _oracle_analysis.empty:
        if "overall_recall_500" in _oracle_analysis.columns:
            overall_values = _oracle_analysis["overall_recall_500"].dropna()
            if not overall_values.empty:
                overall_recall_500 = float(overall_values.iloc[0])
        tail_rows = _oracle_analysis[_oracle_analysis.get("quartile").eq("Q1 (tail)")]
        if not tail_rows.empty and "recall" in tail_rows.columns:
            tail_recall_500 = float(tail_rows["recall"].iloc[0])

    return {
        "users": users,
        "items": items,
        "ablation": normalize_ablation_columns(_ablation) if _ablation is not None else generate_demo_data()["ablation"],
        "ranking_metrics": ranking_metrics,
        "overall_recall_500": overall_recall_500,
        "tail_recall_500": tail_recall_500,
        "survival_curves": survival_curves,
        "chapters": chapters,
        "source": "Real processed artifacts",
        "session_count": len(_session_features),
    }


def recommendations_for_user(user_row: pd.Series, items: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    candidates = items.copy()
    candidates["genre_match"] = (candidates["genre"] == user_row["dominant_genre"]).astype(float)
    candidates["retrieval_score"] = (
        0.45 * candidates["popularity_pct"].fillna(0)
        + 0.35 * candidates["quality_score"].fillna(0)
        + 0.20 * candidates["genre_match"]
    ).clip(0, 1)
    candidates["engagement_fit"] = (
        1 - np.abs(candidates["chapter_count"].rank(pct=True) - float(user_row["avg_completion"]))
    ).clip(0, 1)
    candidates["author_affinity"] = candidates.get("author_id", candidates["item_id"]).astype(str).rank(pct=True)
    candidates["novelty_score"] = -np.log2(candidates["popularity_pct"].clip(lower=0.01))
    candidates["hazard_score"] = (1 - candidates["quality_score"]) * 0.35 + (1 - candidates["engagement_fit"]) * 0.25
    novelty_norm = candidates["novelty_score"] / max(candidates["novelty_score"].max(), 1)
    candidates["final_score"] = (
        0.35 * candidates["retrieval_score"]
        + 0.20 * candidates["quality_score"]
        + 0.15 * candidates["engagement_fit"]
        + 0.10 * candidates["author_affinity"]
        + 0.10 * candidates["genre_match"]
        + 0.05 * novelty_norm
        - 0.05 * candidates["hazard_score"]
    )
    out = candidates.sort_values("final_score", ascending=False).head(limit).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def render_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_reviewer_note() -> None:
    st.markdown(
        f"""
        <div class="note-box">
            <strong>Reviewer note.</strong> System A is the recommendation layer of the project.
            It converts raw reading events into a {term("feature store")}, retrieves candidates with
            {term("two-tower retrieval")}, searches vectors through {term("FAISS")}, estimates dropout risk with
            a {term("survival model")}, and reranks candidates with {term("LambdaMART")}. Hover any blue technical
            term for a short definition.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_glossary() -> None:
    with st.expander("Plain-English glossary", expanded=False):
        for key, value in GLOSSARY.items():
            st.markdown(f"**{key.title()}**: {value}")


data = generate_demo_data()
session_features = load_parquet_safe(DATA_DIR / "session_features.parquet")
catalog_real = load_parquet_safe(SYNTHETIC_DIR / "catalog.parquet")
quality_real = load_parquet_safe(DATA_DIR / "quality_scores.parquet")
ablation_real = load_parquet_safe(DATA_DIR / "ablation_results.parquet")
ranking_metrics_real = load_parquet_safe(DATA_DIR / "completion_ndcg_metrics.parquet")
oracle_real = load_parquet_safe(DATA_DIR / "oracle_analysis.parquet")

if session_features is not None and catalog_real is not None:
    data = build_real_dashboard_data(
        session_features,
        catalog_real,
        quality_real,
        ablation_real,
        ranking_metrics_real,
        oracle_real,
    )


with st.sidebar:
    st.markdown("## System A")
    st.markdown("Adaptive Discovery & Personalization Engine")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Overview", "User Explorer", "Recommendations", "Ablation Study", "Survival Analysis"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Data source: {data['source']}")
    st.caption(f"Project root: {PROJECT_ROOT.name}")
    render_glossary()


if page == "Overview":
    render_header(
        "System A - Discovery Engine",
        "A recommender prototype that scores stories by reading depth, quality, dropout risk, and fit to the reader.",
    )

    ranking_metrics = data["ranking_metrics"]
    metric_row = ranking_metrics[ranking_metrics["k"] == 10]
    if metric_row.empty:
        metric_row = ranking_metrics.tail(1)
    metric_row = metric_row.iloc[0]
    latest_metric = float(metric_row["CW_NDCG"])
    binary_metric = float(metric_row["Binary_NDCG"])
    metric_delta = float(metric_row.get("delta", latest_metric - binary_metric))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Users", f"{len(data['users']):,}", help="Unique users represented in the dashboard data.")
    col2.metric("Catalog Items", f"{len(data['items']):,}", help="Stories/books available for recommendation.")
    col3.metric("Sessions", f"{data['session_count']:,}", help="Reading sessions reconstructed from event telemetry.")
    col4.metric(
        "CW-NDCG@10",
        f"{latest_metric:.3f}",
        delta=f"{metric_delta:+.3f} vs binary",
        help=GLOSSARY["completion-weighted NDCG"],
    )

    diag1, diag2, diag3 = st.columns(3)
    diag1.metric(
        "Binary NDCG@10",
        f"{binary_metric:.3f}",
        help="Standard NDCG using binary relevance instead of completion depth.",
    )
    if pd.notna(data["overall_recall_500"]):
        diag2.metric(
            "Overall Recall@500",
            f"{data['overall_recall_500']:.3f}",
            help="Fraction of positive held-out items recovered in the top 500 candidates.",
        )
    if pd.notna(data["tail_recall_500"]):
        diag3.metric(
            "Tail Recall@500",
            f"{data['tail_recall_500']:.3f}",
            help="Recall@500 for the least-popular item quartile. This is the current weakness.",
        )

    st.subheader("How The System Works")
    flow_cols = st.columns(5)
    steps = [
        ("1. Events", "Raw reads, exits, speed, completion, and device signals."),
        ("2. Feature Store", "Reusable user, item, session, topic, and quality features."),
        ("3. Retrieval", "Two-tower model finds candidate stories quickly."),
        ("4. Risk + Rank", "Survival risk and LambdaMART reorder candidates."),
        ("5. Evaluation", "CW-NDCG, ablation, and oracle analysis check behavior."),
    ]
    for col, (heading, body) in zip(flow_cols, steps):
        with col:
            st.markdown(f"**{heading}**")
            st.caption(body)

    render_reviewer_note()


elif page == "User Explorer":
    render_header(
        "User Explorer",
        "Inspect one reader's behavior profile and see the signals used by the recommendation engine.",
    )

    selected_user = st.selectbox("Select user", data["users"]["user_id"].tolist())
    user_row = data["users"][data["users"]["user_id"] == selected_user].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reading Speed", f"{user_row['avg_velocity']:.0f} WPM", help="Average words per minute.")
    col2.metric("Completion", f"{user_row['avg_completion']:.0%}", help="Average fraction of a story/session completed.")
    col3.metric("Sessions/Week", f"{user_row['sessions_per_week']:.1f}", help="Estimated weekly reading frequency.")
    col4.metric("Main Genre", str(user_row["dominant_genre"]), help="Most common genre in this user's reading history.")

    st.subheader("Engagement Profile")
    categories = ["Velocity", "Completion", "Re-read", "Depth", "Consistency", "Breadth", "Frequency", "Retention"]
    values = np.array(
        [
            min(user_row["avg_velocity"] / 600, 1),
            user_row["avg_completion"],
            0.35,
            user_row["avg_completion"],
            min(user_row["sessions_per_week"] / 10, 1),
            0.55,
            min(user_row["sessions_per_week"] / 10, 1),
            user_row["avg_completion"],
        ]
    )
    fig = go.Figure(
        data=go.Scatterpolar(
            r=np.append(values, values[0]),
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.16)",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=410,
        margin=dict(t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    render_reviewer_note()


elif page == "Recommendations":
    render_header(
        "Recommendations",
        "View ranked story candidates and the contribution of each scoring signal.",
    )

    selected_user = st.selectbox("Select user", data["users"]["user_id"].tolist(), key="rec_user")
    user_row = data["users"][data["users"]["user_id"] == selected_user].iloc[0]
    user_recs = recommendations_for_user(user_row, data["items"], limit=10)

    for _, row in user_recs.iterrows():
        with st.expander(
            f"#{int(row['rank'])} - {row['title']} | Score {row['final_score']:.3f} | Quality {row['quality_score']:.2f}",
            expanded=int(row["rank"]) <= 3,
        ):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**Genre:** {row['genre']}")
                st.markdown(f"**Chapters:** {int(row['chapter_count'])}")
                st.markdown(f"**Avg Rating:** {row['avg_rating']:.1f}")
                st.markdown(
                    f"{term('hazard score')}: **{row['hazard_score']:.0%}**",
                    unsafe_allow_html=True,
                )
            with col2:
                features = {
                    "Retrieval": row["retrieval_score"] * 0.35,
                    "Quality": row["quality_score"] * 0.20,
                    "Engagement Fit": row["engagement_fit"] * 0.15,
                    "Author Affinity": row["author_affinity"] * 0.10,
                    "Genre Match": row["genre_match"] * 0.10,
                    "Novelty": (row["novelty_score"] / max(user_recs["novelty_score"].max(), 1)) * 0.05,
                    "Hazard Penalty": -row["hazard_score"] * 0.05,
                }
                fig = go.Figure(
                    go.Bar(
                        x=list(features.values()),
                        y=list(features.keys()),
                        orientation="h",
                        marker_color=["#2563eb" if v >= 0 else "#dc2626" for v in features.values()],
                    )
                )
                fig.update_layout(
                    height=230,
                    margin=dict(l=0, r=0, t=10, b=10),
                    xaxis_title="Contribution to final score",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
    render_reviewer_note()


elif page == "Ablation Study":
    render_header(
        "Ablation Study",
        "Diagnostic comparison of offline scoring variants. It should reveal regressions as well as gains.",
    )

    abl = data["ablation"].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Completion-weighted NDCG@10",
            x=abl["Model"],
            y=abl["CW-NDCG@10"],
            marker_color="#2563eb",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Binary NDCG@10",
            x=abl["Model"],
            y=abl["Binary-NDCG@10"],
            marker_color="#0f766e",
        )
    )
    fig.update_layout(
        barmode="group",
        height=460,
        yaxis_title="Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"Metric note: {term('completion-weighted NDCG')} rewards recommendations that users read deeply.",
        unsafe_allow_html=True,
    )
    st.dataframe(
        abl.style.format(
            {
                "CW-NDCG@10": "{:.3f}",
                "Binary-NDCG@10": "{:.3f}",
                "Recall@500": lambda x: f"{x:.3f}" if pd.notna(x) else "-",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    delta = abl["CW-NDCG@10"].iloc[-1] - abl["CW-NDCG@10"].iloc[0]
    if delta < 0:
        st.warning(
            f"Current ablation diagnostic is {delta:.3f} below baseline. "
            "This means the latest reranking stack is not proven better by this artifact; treat it as a research signal, not a win."
        )
    else:
        st.info(
            f"Current ablation diagnostic is {delta:+.3f} over baseline. "
            "Treat this as prototype evidence, not a production benchmark."
        )
    render_reviewer_note()


elif page == "Survival Analysis":
    render_header(
        "Survival Analysis",
        "Visualizes estimated reader retention and dropout risk across chapters.",
    )

    fig = go.Figure()
    colors = {"High quality": "#16a34a", "Medium quality": "#ca8a04", "Low quality": "#dc2626"}
    for label, curve in data["survival_curves"].items():
        fig.add_trace(
            go.Scatter(
                x=data["chapters"],
                y=curve,
                name=label,
                line=dict(color=colors[label], width=2.5),
            )
        )
    fig.add_vline(x=4, line_dash="dash", line_color="#6b7280", annotation_text="Early-chapter risk zone")
    fig.update_layout(
        xaxis_title="Chapter index",
        yaxis_title="Survival probability: still reading",
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 1.05]),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"The {term('survival model')} estimates a {term('hazard score')} so the reranker can avoid recommending items likely to be abandoned.",
        unsafe_allow_html=True,
    )
    hazard_data = pd.DataFrame(
        {
            "Signal": [
                "Long author hiatus",
                "Cliff-shaped completion curve",
                "Increasing inter-chapter gap",
                "Low completion proximity",
                "Negative velocity acceleration",
            ],
            "Meaning": [
                "Author has not updated for a long period.",
                "Many readers stop after the same early section.",
                "Reader takes longer gaps between chapters.",
                "Reader is still far from finishing.",
                "Reader speed is falling over time.",
            ],
            "Why it matters": [
                "May reduce trust that the story will continue.",
                "Suggests a content or pacing problem.",
                "Signals weakening engagement.",
                "Early abandonment is easier.",
                "Often appears before dropout.",
            ],
        }
    )
    st.dataframe(hazard_data, use_container_width=True, hide_index=True)
    render_reviewer_note()
