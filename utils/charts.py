"""
All reusable Plotly chart functions for the dashboard.
Each function returns a go.Figure with consistent dark theming.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

# ── THEME ──────────────────────────────────────────────────────────────────
BG_DARK  = "#0a0e17"
BG_PANEL = "#111827"
BORDER   = "#1e2d3d"
GOLD     = "#f0b429"
TEAL     = "#00d9b8"
BLUE     = "#3b82f6"
GREEN    = "#22c55e"
ORANGE   = "#f97316"
RED      = "#ef4444"
SLATE    = "#94a3b8"
PLATINUM = "#e2e8f0"
WHITE    = "#ffffff"
PALETTE  = [BLUE, TEAL, GREEN, ORANGE, RED, GOLD, "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"]

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=BG_PANEL,
    font=dict(family="Inter, Arial, sans-serif", color=PLATINUM, size=12),
    margin=dict(t=50, b=40, l=50, r=30),
    legend=dict(
        bgcolor="rgba(17,24,39,0.8)",
        bordercolor=BORDER, borderwidth=1,
        font=dict(size=11, color=PLATINUM),
    ),
    xaxis=dict(
        showgrid=True, gridcolor="rgba(48,54,61,0.5)", zeroline=False,
        linecolor=BORDER, tickfont=dict(size=10, color=SLATE),
    ),
    yaxis=dict(
        showgrid=True, gridcolor="rgba(48,54,61,0.5)", zeroline=False,
        linecolor=BORDER, tickfont=dict(size=10, color=SLATE),
    ),
)


def apply_theme(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    layout = dict(**LAYOUT_BASE, height=height)
    if title:
        layout["title"] = dict(text=title, font=dict(color=GOLD, size=15), x=0.02)
    fig.update_layout(**layout)
    return fig


# ── EDA CHARTS ─────────────────────────────────────────────────────────────

def plot_annual_trend(df: pd.DataFrame) -> go.Figure:
    """Annual report volume + fatality rate dual-axis."""
    annual = df.groupby("year").agg(
        total=("report_id", "count"),
        serious=("is_fatal", "sum"),
    ).reset_index()
    annual["serious_pct"] = annual["serious"] / annual["total"] * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=annual["year"], y=annual["total"],
        name="Total Reports", marker_color=BLUE,
        marker_line_color=BG_DARK, marker_line_width=0.5,
        hovertemplate="%{x}: %{y:,} reports<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=annual["year"], y=annual["serious_pct"],
        name="Fatality %", line=dict(color=RED, width=3),
        mode="lines+markers", marker=dict(size=8),
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)
    fig.update_yaxes(title_text="Report Count", secondary_y=False, title_font=dict(color=BLUE))
    fig.update_yaxes(title_text="Fatality Rate (%)", secondary_y=True, title_font=dict(color=RED))
    return apply_theme(fig, "📈 Annual Report Volume & Fatality Rate", 380)


def plot_top_drugs(df: pd.DataFrame, n: int = 10) -> go.Figure:
    """Top N drugs by report count."""
    top = df["suspect_drug"].value_counts().head(n).reset_index()
    top.columns = ["drug", "count"]
    colors = [RED if i == 0 else ORANGE if i == 1 else TEAL for i in range(len(top))]
    fig = go.Figure(go.Bar(
        x=top["count"][::-1], y=top["drug"][::-1],
        orientation="h", marker=dict(color=colors[::-1]),
        hovertemplate="%{y}: %{x:,}<extra></extra>",
    ))
    return apply_theme(fig, f"💊 Top {n} Suspect Drugs by Reports", 400)


def plot_top_reactions(df: pd.DataFrame, n: int = 10) -> go.Figure:
    top = df["primary_reaction"].value_counts().head(n).reset_index()
    top.columns = ["reaction", "count"]
    colors = [RED, ORANGE] + [TEAL] * (n - 2)
    fig = go.Figure(go.Bar(
        x=top["count"][::-1], y=top["reaction"][::-1],
        orientation="h", marker=dict(color=colors[::-1]),
        hovertemplate="%{y}: %{x:,}<extra></extra>",
    ))
    return apply_theme(fig, "⚕️ Top Adverse Reactions", 400)


def plot_age_fatality(df: pd.DataFrame) -> go.Figure:
    # Detect actual age-group labels in the data
    all_groups = df["age_group"].dropna().unique().tolist()
    # Canonical ordering — try both naming conventions
    age_order_candidates = [
        ["Pediatric(0-18)", "Adult(19-40)", "Middle-Aged(41-60)", "Senior(61-80)", "Elderly(81+)"],
        ["Pediatric(0-18)", "Adult(19-40)", "Middle-Aged(41-65)", "Senior(66-80)", "Elderly(81+)"],
    ]
    age_order = None
    for candidate in age_order_candidates:
        if set(candidate).issubset(set(all_groups)):
            age_order = candidate
            break
    if age_order is None:
        # Fallback: just use whatever groups exist, sorted
        age_order = sorted([g for g in all_groups if g and g != "Unknown"])

    ag = df[df["age_group"].isin(age_order)].groupby("age_group")["is_fatal"].mean() * 100
    ag = ag.reindex(age_order).reset_index()
    ag.columns = ["age_group", "fatal_pct"]
    ag = ag.dropna()
    bar_colors = [GREEN if v < 5 else ORANGE if v < 15 else RED for v in ag["fatal_pct"]]
    fig = go.Figure(go.Bar(
        x=ag["age_group"], y=ag["fatal_pct"],
        marker=dict(color=bar_colors),
        text=[f"{v:.1f}%" for v in ag["fatal_pct"]],
        textposition="outside", textfont=dict(color=WHITE),
        hovertemplate="%{x}: %{y:.1f}% fatality<extra></extra>",
    ))
    return apply_theme(fig, "👥 Fatality Rate by Age Group", 380)


def plot_sex_distribution(df: pd.DataFrame) -> go.Figure:
    sex = df[df["patient_sex"] != "Unknown"]["patient_sex"].value_counts()
    fig = go.Figure(go.Pie(
        labels=sex.index, values=sex.values,
        hole=0.6,
        marker=dict(colors=[BLUE, RED], line=dict(color=BG_DARK, width=2)),
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
        textfont=dict(color=WHITE, size=13),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG_PANEL,
        font=dict(color=PLATINUM), height=320,
        title=dict(text="⚧ Sex Distribution", font=dict(color=GOLD, size=14), x=0.02),
        legend=dict(bgcolor="rgba(17,24,39,0.8)", bordercolor=BORDER),
    )
    return fig


def plot_quarterly_trend(df: pd.DataFrame) -> go.Figure:
    quarterly = df.groupby("quarter").agg(
        total=("report_id", "count"),
        fatal=("is_fatal", "sum"),
    ).reset_index().sort_values("quarter")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=quarterly["quarter"], y=quarterly["total"],
        fill="tozeroy", fillcolor="rgba(59,130,246,0.15)",
        line=dict(color=BLUE, width=2), name="Total Reports",
    ))
    fig.add_trace(go.Scatter(
        x=quarterly["quarter"], y=quarterly["fatal"],
        fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
        line=dict(color=RED, width=2), name="Fatal Reports",
    ))
    fig = apply_theme(fig, "📅 Quarterly Reporting Trend 2015–2025", 320)
    fig.update_xaxes(tickangle=45, nticks=10)
    return fig


def plot_polypharmacy_fatal(df: pd.DataFrame) -> go.Figure:
    poly_order = ["Single", "2-3 drugs", "4-5 drugs", "Polypharmacy(6+)"]
    pg = df[df["drug_count_category"].isin(poly_order)].groupby("drug_count_category")["is_fatal"].mean() * 100
    pg = pg.reindex(poly_order).reset_index()
    pg.columns = ["category", "fatal_pct"]
    pg = pg.dropna()
    colors = [GREEN, TEAL, ORANGE, RED][:len(pg)]
    fig = go.Figure(go.Bar(
        x=pg["category"], y=pg["fatal_pct"],
        marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in pg["fatal_pct"]],
        textposition="outside", textfont=dict(color=WHITE),
    ))
    return apply_theme(fig, "💉 Fatality Rate by Polypharmacy", 340)


def plot_outcome_rates(df: pd.DataFrame) -> go.Figure:
    outcomes = {
        "Fatal": df["is_fatal"].mean() * 100,
        "Hospitalized": df["is_hospitalized"].mean() * 100,
        "Life-Threat": df["is_life_threat"].mean() * 100,
        "Disabling": df["is_disabling"].mean() * 100,
    }
    fig = go.Figure(go.Bar(
        x=list(outcomes.keys()), y=list(outcomes.values()),
        marker=dict(color=[RED, ORANGE, "#8b5cf6", TEAL]),
        text=[f"{v:.1f}%" for v in outcomes.values()],
        textposition="outside", textfont=dict(color=WHITE),
    ))
    return apply_theme(fig, "🚨 Serious Outcome Rates (%)", 320)


def plot_country_top(df: pd.DataFrame, n: int = 10) -> go.Figure:
    top_c = df[df["country"] != "Unknown"]["country"].value_counts().head(n).reset_index()
    top_c.columns = ["country", "count"]
    fig = go.Figure(go.Bar(
        x=top_c["count"][::-1], y=top_c["country"][::-1],
        orientation="h", marker=dict(color=TEAL),
        hovertemplate="%{y}: %{x:,}<extra></extra>",
    ))
    return apply_theme(fig, f"🌍 Top {n} Reporting Countries", 380)


def plot_route_distribution(df: pd.DataFrame) -> go.Figure:
    route = df["drug_route"].value_counts().head(8).reset_index()
    route.columns = ["route", "count"]
    fig = go.Figure(go.Pie(
        labels=route["route"], values=route["count"],
        hole=0.5,
        marker=dict(colors=PALETTE, line=dict(color=BG_DARK, width=1)),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", height=350,
        title=dict(text="💊 Drug Administration Routes", font=dict(color=GOLD, size=14), x=0.02),
        font=dict(color=PLATINUM),
        legend=dict(bgcolor="rgba(17,24,39,0.8)", bordercolor=BORDER),
    )
    return fig


# ── MODEL PERFORMANCE CHARTS ────────────────────────────────────────────────

def plot_roc_curve(fpr, tpr, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        line=dict(color=GREEN, width=3), mode="lines",
        fill="tozeroy", fillcolor="rgba(34,197,94,0.1)",
        name=f"LightGBM (AUC = {auc:.4f})",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(color=SLATE, width=1.5, dash="dash"),
        mode="lines", name="Random Baseline",
    ))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return apply_theme(fig, "📊 ROC Curve — Fatal Outcome Prediction", 400)


def plot_feature_importance(feat_imp: pd.DataFrame) -> go.Figure:
    fi = feat_imp.sort_values("importance").tail(12)
    imp_max = fi["importance"].max()
    colors = [
        RED if v / imp_max > 0.7 else ORANGE if v / imp_max > 0.4 else TEAL
        for v in fi["importance"]
    ]
    fig = go.Figure(go.Bar(
        x=fi["importance"], y=fi["feature"],
        orientation="h", marker=dict(color=colors),
        hovertemplate="%{y}: %{x:,}<extra></extra>",
    ))
    return apply_theme(fig, "🤖 Feature Importance — LightGBM", 400)


def plot_confusion_matrix(cm: list) -> go.Figure:
    labels = ["Not Fatal", "Fatal"]
    z = [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]]
    text = [[f"{v:,}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale=[[0, BG_PANEL], [1, "#ef4444"]],
        text=text, texttemplate="%{text}",
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
        textfont=dict(size=18, color=WHITE),
        showscale=False,
    ))
    fig.update_yaxes(title="Actual Label")
    fig.update_xaxes(title="Predicted Label")
    return apply_theme(fig, "🧮 Confusion Matrix", 350)


def plot_risk_gauge(probability: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(probability * 100, 1),
        number=dict(suffix="%", font=dict(size=40, color=PLATINUM)),
        delta=dict(reference=10.3, increasing=dict(color=RED), decreasing=dict(color=GREEN)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=SLATE, tickfont=dict(color=SLATE)),
            bar=dict(
                color=RED if probability >= 0.7 else ORANGE if probability >= 0.4 else GREEN,
                thickness=0.3,
            ),
            bgcolor=BG_PANEL, bordercolor=BORDER,
            steps=[
                dict(range=[0, 40], color="rgba(34,197,94,0.1)"),
                dict(range=[40, 70], color="rgba(249,115,22,0.1)"),
                dict(range=[70, 100], color="rgba(239,68,68,0.1)"),
            ],
            threshold=dict(line=dict(color=GOLD, width=4), thickness=0.75, value=10.3),
        ),
        title=dict(text="Fatality Risk Score", font=dict(color=SLATE, size=14)),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=PLATINUM),
        height=280, margin=dict(t=30, b=10),
    )
    return fig
