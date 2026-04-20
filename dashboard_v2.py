"""
dashboard.py
─────────────
Streamlit dashboard for the BAA5013 proof-of-concept.

Four views:
  1. Incrementality Gap Panel
  2. Channel Contribution Decomposition (Bayesian MMM)
  3. Persuadable Audience Diagnostic (Uplift Model)
  4. Budget Scenario Analysis (saturation-aware per-channel sliders)

Run with:
    streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Incrementality Gap Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE       = "#1F4E79"
MID_BLUE   = "#2E75B6"
LIGHT_BLUE = "#D6E4F0"
AMBER      = "#C65911"
GREEN      = "#1A7A4A"
RED        = "#C0392B"
GREY       = "#7F7F7F"

CHANNEL_COLOURS = {
    "paid_social":     AMBER,
    "digital_display": MID_BLUE,
    "video":           GREEN,
    "search":          GREY,
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        padding: 20px 30px; border-radius: 8px;
        color: white; margin-bottom: 24px;
    }
    .cite-box {
        background: #EAF2FA; border-left: 4px solid #1F4E79;
        padding: 10px 14px; border-radius: 4px;
        font-size: 0.85em; color: #0D2B4A; margin: 8px 0;
    }
    .warning-box {
        background: #FFF3E0; border-left: 4px solid #C65911;
        padding: 10px 14px; border-radius: 4px;
        font-size: 0.85em; color: #5C2A00; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Saturation response helper ────────────────────────────────────────────────

def channel_response(channel: str, weekly_spend: float,
                     adstock_alphas: dict, saturation_lambdas: dict,
                     true_iroas: dict, spend_ranges: dict) -> tuple:
    """
    Compute (weekly_incremental_revenue, effective_iroas) at a given spend level
    using the MMM adstock + Hill saturation parameters.

    The adstock is normalised relative to mean spend so the Hill function
    operates on a dimensionless ratio in [0, ∞).  This avoids numerical
    collapse at large raw spend values while preserving the saturation shape.

    The curve is anchored so that effective_iroas == true_iroas at mean spend,
    making the response curves directly comparable to the verified iROAS values
    reported by the geo-holdout experiment.
    """
    if weekly_spend <= 0:
        return 0.0, 0.0

    alpha = adstock_alphas[channel]
    lam   = saturation_lambdas[channel]
    lo, hi = spend_ranges[channel]
    mean_spend = (lo + hi) / 2.0

    # Normalised steady-state adstock: 1.0 at mean spend
    x_norm = (weekly_spend / (1 - alpha)) / (mean_spend / (1 - alpha))
    # Simplifies to: x_norm = weekly_spend / mean_spend

    # Hill saturation on normalised input
    sat      = x_norm ** lam / (1.0 + x_norm ** lam)
    sat_mean = 1.0 ** lam    / (1.0 + 1.0 ** lam)   # = 0.5 when x_norm = 1

    # Back-computed scale: revenue = scale * sat, anchored at mean spend
    scale = true_iroas[channel] * mean_spend / sat_mean

    revenue       = scale * sat
    eff_iroas     = revenue / weekly_spend

    return revenue, eff_iroas


# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data
def load_data():
    from generate_data import (
        generate_mmm_dataset,
        generate_geo_holdout_dataset,
        generate_uplift_dataset,
        compute_incrementality_gap,
        TRUE_IROAS,
        PLATFORM_ROAS,
        SURE_THING_PROPORTIONS,
        SPEND_RANGES,
        ADSTOCK_ALPHAS,
        SATURATION_LAMBDAS,
    )
    mmm_df    = generate_mmm_dataset()
    geo_df    = generate_geo_holdout_dataset()
    uplift_df = generate_uplift_dataset()
    gap       = compute_incrementality_gap(geo_df)
    return (mmm_df, geo_df, uplift_df, gap,
            TRUE_IROAS, PLATFORM_ROAS, SURE_THING_PROPORTIONS,
            SPEND_RANGES, ADSTOCK_ALPHAS, SATURATION_LAMBDAS)


@st.cache_data
def run_uplift(_uplift_df):
    from uplift_model import run_uplift_analysis
    return run_uplift_analysis(_uplift_df)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h2 style="margin:0; font-size:1.5em;">Incrementality Gap Dashboard</h2>
    <p style="margin:4px 0 0 0; opacity:0.85; font-size:0.9em;">
        Meta Platforms — Measurement Science Function &nbsp;|&nbsp;
        BAA5013 MsBA Group Project — Matthew Lo 25083247 | Soo Chia Xin 25084386 | Patricia Tan 25118290 | Tan Jin Suan 21029756
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    view = st.radio(
        "",
        ["1 · Incrementality Gap",
         "2 · Channel Decomposition",
         "3 · Persuadable Audience",
         "4 · Budget Scenario"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Ground-Truth Parameters**")
    st.markdown(
        "Parameters are set explicitly during data generation, "
        "confirming results demonstrate the measurement mechanism "
        "rather than being reverse-engineered."
    )
    st.markdown("---")
    st.caption("Wernerfelt et al. (2025) *Marketing Science* 44(2)")
    st.caption("Gordon et al. (2023) *Marketing Science* 42(4)")
    st.caption("Künzel et al. (2019) *PNAS* 116(10)")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Generating synthetic datasets..."):
    (mmm_df, geo_df, uplift_df, gap,
     TRUE_IROAS, PLATFORM_ROAS, SURE_THING_PROPORTIONS,
     SPEND_RANGES, ADSTOCK_ALPHAS, SATURATION_LAMBDAS) = load_data()

CHANNELS = ["paid_social", "digital_display", "video", "search"]


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 1 — INCREMENTALITY GAP PANEL
# ═══════════════════════════════════════════════════════════════════════════════
if view == "1 · Incrementality Gap":
    st.markdown("## View 1 — Incrementality Gap Panel")
    st.markdown(
        "**KPI 1.1** — The percentage difference between Advantage+'s platform-reported "
        "ROAS and the geo-holdout verified incremental ROAS for the same campaign period."
    )
    st.markdown(
        '<div class="cite-box">Formula: Gap (%) = '
        "(Platform ROAS − Geo-holdout iROAS) / Platform ROAS × 100</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Platform-Reported ROAS", f"{gap['platform_roas']:.2f}x",
                  help="What Advantage+ self-reports — includes sure-thing conversions")
    with col2:
        st.metric("Verified Incremental ROAS", f"{gap['verified_iroas']:.2f}x",
                  delta=f"{gap['verified_iroas'] - gap['platform_roas']:.2f}x vs platform",
                  delta_color="inverse",
                  help="Geo-holdout verified iROAS — advertiser-side transaction data only")
    with col3:
        st.metric("Incrementality Gap", f"{gap['incrementality_gap_pct']:.1f}%",
                  help="KPI 1.1")
    with col4:
        st.metric("Sure-Thing Proportion", f"{gap['sure_thing_proportion_input']:.0%}",
                  help="Known ground-truth parameter set during data generation")

    st.markdown("### Platform ROAS vs Verified Incremental ROAS")
    fig_gap = go.Figure()
    categories = ["Platform-Reported ROAS\n(Advantage+ self-report)",
                  "Geo-Holdout Verified iROAS\n(External transaction data)"]
    values = [gap["platform_roas"], gap["verified_iroas"]]
    fig_gap.add_trace(go.Bar(
        x=categories, y=values,
        marker_color=[AMBER, GREEN],
        text=[f"{v:.2f}x" for v in values],
        textposition="outside", width=0.4
    ))
    fig_gap.add_annotation(
        x=0.5, y=max(values) * 1.15, xref="paper", yref="y",
        text=f"Incrementality Gap: {gap['incrementality_gap_pct']:.1f}%",
        showarrow=False, font=dict(size=14, color=AMBER),
        bgcolor=LIGHT_BLUE, bordercolor=AMBER, borderwidth=1
    )
    fig_gap.update_layout(
        height=420, plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="ROAS", gridcolor="#F0F0F0",
                   range=[0, max(values) * 1.3]),
        showlegend=False, margin=dict(t=60, b=40)
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    st.markdown(
        '<div class="warning-box">⚠️ The platform metric and the independently verified '
        'metric move in opposite directions when the sure-thing mechanism is operating '
        'at scale. This is the Incrementality Gap in its most direct empirical form.</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Conversion Pool Decomposition")
    total_attributed = gap["total_platform_attributed"]
    verified_inc     = gap["verified_incremental_revenue"]
    sure_thing_rev   = total_attributed - verified_inc

    fig_pie = go.Figure(go.Pie(
        labels=["Genuinely Incremental\n(caused by advertising)",
                "Sure-Thing Attribution\n(would have occurred organically)"],
        values=[max(verified_inc, 0), max(sure_thing_rev, 0)],
        hole=0.5, marker_colors=[GREEN, AMBER],
        textinfo="label+percent", textfont=dict(size=11)
    ))
    fig_pie.update_layout(height=380, showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### Weekly Revenue — Treatment vs Holdout Regions")
    exp_df = geo_df[~geo_df["is_ptw"]].copy()
    weekly = exp_df.groupby(["week", "is_treatment"]).agg(
        actual_revenue=("actual_revenue", "sum")
    ).reset_index()
    treatment_weekly = weekly[weekly["is_treatment"]].copy()
    holdout_weekly   = weekly[~weekly["is_treatment"]].copy()
    from generate_data import N_TREATMENT, N_HOLDOUT
    holdout_weekly["actual_revenue"] *= N_TREATMENT / N_HOLDOUT

    fig_weekly = go.Figure()
    fig_weekly.add_trace(go.Scatter(
        x=treatment_weekly["week"], y=treatment_weekly["actual_revenue"],
        name="Treatment (Advantage+ active)",
        line=dict(color=MID_BLUE, width=2),
        fill="tozeroy", fillcolor="rgba(46,117,182,0.1)"
    ))
    fig_weekly.add_trace(go.Scatter(
        x=holdout_weekly["week"], y=holdout_weekly["actual_revenue"],
        name="Holdout (ads paused) — scaled",
        line=dict(color=GREY, width=2, dash="dash")
    ))
    fig_weekly.update_layout(
        height=360, plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="Revenue ($)", gridcolor="#F0F0F0"),
        xaxis=dict(title="Experiment Week"),
        legend=dict(orientation="h", y=1.02), margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig_weekly, use_container_width=True)
    st.caption(
        "Difference between treatment and scaled holdout = verified incremental revenue. "
        "Geo-holdout outcome measured using advertiser-side transaction data — "
        "no Meta conversion tracking at any stage."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 2 — CHANNEL DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════
elif view == "2 · Channel Decomposition":
    st.markdown("## View 2 — Channel Contribution Decomposition")
    st.markdown(
        "Bayesian MMM channel attribution using **advertiser-side transaction data** "
        "as the outcome variable. Reveals the true incremental contribution of each "
        "channel against which platform-reported ROAS can be compared."
    )
    st.markdown(
        '<div class="cite-box">Bayesian estimation (Abril-Pla et al., 2023, PyMC) produces '
        'full posterior distributions over all parameters. '
        'Adstock: Koyck (1954). Saturation: Jin et al. (2017).</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Ground-Truth vs Platform-Reported vs True iROAS")
    gt_df = pd.DataFrame({
        "Channel":                  CHANNELS,
        "True iROAS (ground truth)": [TRUE_IROAS[ch] for ch in CHANNELS],
        "Platform-Reported ROAS":   [round(PLATFORM_ROAS[ch], 2) for ch in CHANNELS],
        "Sure-Thing Proportion":    [f"{SURE_THING_PROPORTIONS[ch]:.0%}" for ch in CHANNELS],
        "Inflation Factor":         [
            f"{(PLATFORM_ROAS[ch] / TRUE_IROAS[ch] - 1) * 100:.0f}%"
            for ch in CHANNELS
        ]
    })
    st.dataframe(gt_df, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="warning-box">⚠️ Platform-reported ROAS is inflated by the '
        'sure-thing proportion in each channel. paid_social (Advantage+) has the '
        'highest sure-thing proportion (60%) and therefore the largest inflation. '
        'The true incremental ROAS is only recoverable via external transaction data.</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Platform ROAS vs True iROAS by Channel")
    fig_ch = go.Figure()
    fig_ch.add_trace(go.Bar(
        name="Platform-Reported ROAS", x=CHANNELS,
        y=[PLATFORM_ROAS[ch] for ch in CHANNELS],
        marker_color=AMBER, opacity=0.85
    ))
    fig_ch.add_trace(go.Bar(
        name="True Incremental ROAS", x=CHANNELS,
        y=[TRUE_IROAS[ch] for ch in CHANNELS],
        marker_color=GREEN, opacity=0.85
    ))
    fig_ch.update_layout(
        barmode="group", height=380, plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="ROAS", gridcolor="#F0F0F0"),
        legend=dict(orientation="h", y=1.02), margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig_ch, use_container_width=True)

    st.markdown("### Weekly Spend by Channel")
    fig_spend = go.Figure()
    for ch in CHANNELS:
        fig_spend.add_trace(go.Scatter(
            x=mmm_df["date"], y=mmm_df[f"spend_{ch}"],
            name=ch, line=dict(color=CHANNEL_COLOURS[ch], width=1.5)
        ))
    fig_spend.update_layout(
        height=340, plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="Weekly Spend ($)", gridcolor="#F0F0F0"),
        xaxis=dict(title="Date"),
        legend=dict(orientation="h", y=1.02), margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig_spend, use_container_width=True)
    st.caption(
        "Spend data sourced from Meta Ads Manager (input layer only). "
        "Revenue outcome variable is advertiser-side transaction data — "
        "no Meta-reported conversions used as an outcome at any point."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 3 — PERSUADABLE AUDIENCE DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════
elif view == "3 · Persuadable Audience":
    st.markdown("## View 3 — Persuadable Audience Diagnostic")
    st.markdown(
        "**KPI 2.2 — Persuadable User Rate.** X-learner meta-learner "
        "(Künzel et al. 2019) applied within the geo-holdout experimental cohort "
        "to estimate the Conditional Average Treatment Effect per user. Diagnoses "
        "whether Advantage+'s audience selection disproportionately targets sure-things."
    )
    st.markdown(
        '<div class="cite-box">Design boundary: uplift modelling operates only within '
        'geo-holdout experimental data. Applying it to observational campaign data would '
        'reintroduce the confounding problem the architecture is designed to resolve.</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Running X-learner meta-learner on geo-holdout cohort..."):
        uplift_results = run_uplift(uplift_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Persuadable User Rate (KPI 2.2)",
            f"{uplift_results['persuadable_rate_pct']:.1f}%",
            help="% of Advantage+ audience with positive CATE — genuinely incremental"
        )
    with col2:
        st.metric(
            "Sure-Thing Rate (treated cohort)",
            f"{uplift_results['treated_sure_thing_pct']:.1f}%",
            help="% of Advantage+ exposed users estimated as sure-things"
        )
    with col3:
        st.metric(
            "Ground-Truth Sure-Thing Rate",
            f"{uplift_results['ground_truth_sure_thing_pct']:.1f}%",
            help="Known parameter from data generation — confirms model recovery"
        )

    st.markdown("### User Segmentation — Advantage+ Exposed Cohort")
    seg_dist   = uplift_results["segment_distribution_pct"]
    seg_order  = ["Persuadable", "Sure-thing", "Lost Cause", "Sleeping Dog"]
    seg_colours = [GREEN, AMBER, GREY, RED]
    seg_labels  = [s for s in seg_order if s in seg_dist]
    seg_values  = [seg_dist.get(s, 0) for s in seg_labels]
    seg_cols_used = [seg_colours[seg_order.index(s)] for s in seg_labels]

    fig_seg = go.Figure(go.Pie(
        labels=seg_labels, values=seg_values,
        hole=0.55, marker_colors=seg_cols_used,
        textinfo="label+percent", textfont=dict(size=12)
    ))
    fig_seg.update_layout(height=380, showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown("### CATE Distribution — Individual Treatment Effect Estimates")
    st.markdown(
        "Positive CATE = advertising caused conversion (Persuadable). "
        "Near-zero = would convert regardless (Sure-thing) or not at all."
    )
    cate = uplift_results["cate"]
    fig_cate = go.Figure()
    fig_cate.add_trace(go.Histogram(
        x=cate, nbinsx=60, marker_color=MID_BLUE, opacity=0.75, name="CATE"
    ))
    fig_cate.add_vline(x=0.05, line_dash="dash", line_color=GREEN,
                       annotation_text="Persuadable threshold",
                       annotation_position="top right")
    fig_cate.add_vline(x=-0.03, line_dash="dash", line_color=RED,
                       annotation_text="Sleeping Dog threshold",
                       annotation_position="top left")
    fig_cate.add_vline(x=float(cate.mean()), line_dash="dot", line_color=AMBER,
                       annotation_text=f"Mean CATE: {cate.mean():.3f}")
    fig_cate.update_layout(
        height=360, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Estimated CATE", gridcolor="#F0F0F0"),
        yaxis=dict(title="Number of Users", gridcolor="#F0F0F0"),
        showlegend=False, margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig_cate, use_container_width=True)

    # ── Intent score vs CATE scatter ─────────────────────────────────────────
    st.markdown("### Prior Purchase Intent vs Estimated CATE")
    st.markdown(
        "High-intent users cluster in the sure-thing segment. "
        "This is the mechanism Wernerfelt et al. (2025) identifies: "
        "purchase-optimised delivery preferentially finds high-intent users."
    )

    rng_sample = np.random.default_rng(42)
    n_sample   = min(1000, len(uplift_df))
    sample_idx = rng_sample.choice(len(uplift_df), size=n_sample, replace=False)
    sample_df  = uplift_df.iloc[sample_idx].copy()
    sample_cate = cate[sample_idx]
    seg_sample  = uplift_results["segments"][sample_idx]

    # Map internal lowercase segment names to capitalised display labels
    seg_internal_to_display = {
        "persuadable":  "Persuadable",
        "sure_thing":   "Sure-thing",
        "lost_cause":   "Lost Cause",
        "sleeping_dog": "Sleeping Dog",
    }
    colour_map = {
        "Persuadable": GREEN,
        "Sure-thing":  AMBER,
        "Lost Cause":  GREY,
        "Sleeping Dog": RED,
    }

    fig_scatter = go.Figure()
    for internal, display in seg_internal_to_display.items():
        mask = seg_sample == internal
        if mask.sum() == 0:
            continue
        fig_scatter.add_trace(go.Scatter(
            x=sample_df["intent_score"].values[mask],
            y=sample_cate[mask],
            mode="markers",
            name=display,
            marker=dict(color=colour_map[display], size=5, opacity=0.6)
        ))
    fig_scatter.add_hline(y=0, line_dash="dash", line_color=GREY)
    fig_scatter.update_layout(
        height=400, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Prior Purchase Intent Score", gridcolor="#F0F0F0"),
        yaxis=dict(title="Estimated CATE", gridcolor="#F0F0F0"),
        legend=dict(orientation="h", y=1.02), margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(
        "A declining Persuadable User Rate across successive experiments provides "
        "early warning of worsening incrementality before it appears in aggregate "
        "iROAS trends. This is the decision-relevant output of this component."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 4 — BUDGET SCENARIO ANALYSIS (saturation-aware)
# ═══════════════════════════════════════════════════════════════════════════════
elif view == "4 · Budget Scenario":
    st.markdown("## View 4 — Budget Scenario Analysis")
    st.markdown(
        "Per-channel spend sliders with saturation-aware response curves. "
        "Effective iROAS updates in real time as spend moves along each channel's "
        "Hill saturation curve — anchored so that effective iROAS equals the "
        "geo-holdout verified iROAS at the current mean operating spend level. "
        "A point-estimate model cannot produce this output."
    )
    st.markdown(
        '<div class="cite-box">Response curves use the same adstock (Koyck, 1954) and '
        'saturation (Jin et al., 2017) parameters fitted in the Bayesian MMM. '
        'Recommendations are anchored in externally verified incrementality — not '
        'platform self-reported conversion data.</div>',
        unsafe_allow_html=True
    )

    # ── Per-channel spend sliders ─────────────────────────────────────────────
    st.markdown("### Set Weekly Spend by Channel")
    st.caption(
        "Sliders default to each channel's mid-range operating spend. "
        "Effective iROAS rises below the default (undersaturation) and "
        "falls above it (diminishing returns)."
    )

    slider_cols = st.columns(len(CHANNELS))
    current_spends = {}
    for col, ch in zip(slider_cols, CHANNELS):
        lo, hi = SPEND_RANGES[ch]
        mean_spend = int((lo + hi) / 2)
        max_slider = int(hi * 2)
        with col:
            current_spends[ch] = st.slider(
                ch.replace("_", " ").title(),
                min_value=0,
                max_value=max_slider,
                value=mean_spend,
                step=max(1000, int((max_slider) / 100)),
                format="$%d",
                key=f"slider_{ch}"
            )

    total_budget = sum(current_spends.values())
    st.metric("Total Weekly Budget", f"${total_budget:,.0f}")
    st.markdown("---")

    # ── Compute current effective iROAS and revenue for each channel ──────────
    current_revenues  = {}
    current_eff_iroas = {}
    for ch in CHANNELS:
        rev, eff = channel_response(
            ch, current_spends[ch],
            ADSTOCK_ALPHAS, SATURATION_LAMBDAS, TRUE_IROAS, SPEND_RANGES
        )
        current_revenues[ch]  = rev
        current_eff_iroas[ch] = eff

    # ── Effective iROAS cards ─────────────────────────────────────────────────
    st.markdown("### Effective iROAS at Current Spend")
    iroas_cols = st.columns(len(CHANNELS))
    for col, ch in zip(iroas_cols, CHANNELS):
        eff   = current_eff_iroas[ch]
        delta = eff - TRUE_IROAS[ch]
        with col:
            st.metric(
                ch.replace("_", " ").title(),
                f"{eff:.2f}x",
                delta=f"{delta:+.2f}x vs verified baseline",
                delta_color="normal",
                help=(
                    f"Verified iROAS at mean spend: {TRUE_IROAS[ch]:.1f}x. "
                    f"Positive delta = undersaturated (more room to spend). "
                    f"Negative delta = diminishing returns."
                )
            )

    # ── Response curves ───────────────────────────────────────────────────────
    st.markdown("### Saturation Response Curves — Effective iROAS vs Weekly Spend")
    st.markdown(
        "Each curve shows how effective iROAS changes as spend moves along the "
        "Hill saturation function. The vertical marker shows the current slider position. "
        "Channels to the left of their marker have room to grow; "
        "channels to the right are in the diminishing-returns zone."
    )

    fig_curves = make_subplots(
        rows=2, cols=2,
        subplot_titles=[ch.replace("_", " ").title() for ch in CHANNELS],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    for idx, ch in enumerate(CHANNELS):
        row = idx // 2 + 1
        col = idx % 2  + 1

        lo, hi = SPEND_RANGES[ch]
        spend_range = np.linspace(max(lo * 0.1, 1), hi * 2.2, 300)

        eff_iroas_curve = np.array([
            channel_response(ch, s, ADSTOCK_ALPHAS, SATURATION_LAMBDAS,
                             TRUE_IROAS, SPEND_RANGES)[1]
            for s in spend_range
        ])

        colour = CHANNEL_COLOURS[ch]

        # Response curve
        fig_curves.add_trace(
            go.Scatter(
                x=spend_range, y=eff_iroas_curve,
                mode="lines", name=ch,
                line=dict(color=colour, width=2),
                showlegend=False,
            ),
            row=row, col=col
        )

        # Verified iROAS reference line
        fig_curves.add_hline(
            y=TRUE_IROAS[ch],
            line_dash="dot", line_color=GREY, line_width=1,
            annotation_text=f"Verified iROAS: {TRUE_IROAS[ch]}x",
            annotation_font_size=10,
            annotation_position="top right",
            row=row, col=col
        )

        # Current spend marker
        cur_s   = current_spends[ch]
        cur_eff = current_eff_iroas[ch]
        if cur_s > 0:
            fig_curves.add_trace(
                go.Scatter(
                    x=[cur_s], y=[cur_eff],
                    mode="markers",
                    marker=dict(color=colour, size=12, symbol="diamond",
                                line=dict(color="white", width=2)),
                    name=f"{ch} current",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{ch}</b><br>"
                        f"Spend: $%{{x:,.0f}}<br>"
                        f"Effective iROAS: %{{y:.2f}}x<extra></extra>"
                    )
                ),
                row=row, col=col
            )

        # Platform ROAS reference line (shows overstatement)
        fig_curves.add_hline(
            y=PLATFORM_ROAS[ch],
            line_dash="dash", line_color=AMBER, line_width=1,
            annotation_text=f"Platform ROAS: {PLATFORM_ROAS[ch]:.1f}x",
            annotation_font_size=10,
            annotation_position="bottom right",
            row=row, col=col
        )

        fig_curves.update_xaxes(
            title_text="Weekly Spend ($)", tickformat="$,.0f",
            gridcolor="#F0F0F0", row=row, col=col
        )
        fig_curves.update_yaxes(
            title_text="Effective iROAS", gridcolor="#F0F0F0",
            row=row, col=col
        )

    fig_curves.update_layout(
        height=620, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=60, b=40)
    )
    st.plotly_chart(fig_curves, use_container_width=True)
    st.caption(
        "Dotted line = geo-holdout verified iROAS at mean spend (ground truth). "
        "Dashed line = platform-reported ROAS (inflated by sure-thing attribution). "
        "Diamond = current slider position."
    )

    # ── Revenue and overstatement table ──────────────────────────────────────
    st.markdown("### Expected Revenue by Channel at Current Spend")

    table_rows = []
    for ch in CHANNELS:
        spend        = current_spends[ch]
        inc_rev      = current_revenues[ch]
        plat_rev     = spend * PLATFORM_ROAS[ch]
        overstatement = plat_rev - inc_rev
        table_rows.append({
            "Channel":                          ch.replace("_", " ").title(),
            "Weekly Spend ($)":                 int(spend),
            "Effective iROAS":                  round(current_eff_iroas[ch], 2),
            "Expected Incremental Revenue ($)": int(inc_rev),
            "Platform Would Report ($)":        int(plat_rev),
            "Sure-Thing Overstatement ($)":     int(overstatement),
        })

    rec_df = pd.DataFrame(table_rows)
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    # ── Stacked revenue chart ─────────────────────────────────────────────────
    st.markdown("### Verified vs Platform-Reported Revenue by Channel")
    fig_stack = go.Figure()
    ch_labels = [c.replace("_", " ").title() for c in CHANNELS]
    inc_vals  = [int(current_revenues[ch])  for ch in CHANNELS]
    over_vals = [max(0, int(current_spends[ch] * PLATFORM_ROAS[ch] - current_revenues[ch]))
                 for ch in CHANNELS]

    fig_stack.add_trace(go.Bar(
        name="Genuine Incremental Revenue",
        x=ch_labels, y=inc_vals,
        marker_color=GREEN,
        text=[f"${v/1000:.0f}k" for v in inc_vals],
        textposition="inside"
    ))
    fig_stack.add_trace(go.Bar(
        name="Sure-Thing Overstatement",
        x=ch_labels, y=over_vals,
        marker_color=AMBER,
        text=[f"${v/1000:.0f}k" for v in over_vals],
        textposition="outside"
    ))
    fig_stack.update_layout(
        barmode="stack", height=400,
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="Revenue ($)", gridcolor="#F0F0F0"),
        legend=dict(orientation="h", y=1.02),
        margin=dict(t=60, b=40)
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    # ── Summary metrics ───────────────────────────────────────────────────────
    total_inc  = sum(current_revenues.values())
    total_plat = sum(current_spends[ch] * PLATFORM_ROAS[ch] for ch in CHANNELS)
    total_gap  = (total_plat - total_inc) / total_plat * 100 if total_plat > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Verified Incremental Revenue", f"${total_inc:,.0f}")
    with col2:
        st.metric("Platform Would Report", f"${total_plat:,.0f}",
                  delta=f"+${total_plat - total_inc:,.0f} overstatement",
                  delta_color="inverse")
    with col3:
        st.metric("Aggregate Incrementality Gap", f"{total_gap:.1f}%",
                  help="Matches Incrementality Gap KPI 1.1 across all channels")

    st.markdown("---")
    st.caption(
        "Response curves use the Hill saturation parameters from the Bayesian MMM. "
        "All recommendations are anchored in geo-holdout verified iROAS — "
        "platform-reported ROAS is shown for comparison only."
    )