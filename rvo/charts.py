"""Plotly figures for the Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .crossover import find_crossings
from .model import Result

OWNER = "#1f77b4"
OWNER_NET = "#6baed6"
RENTER = "#ff7f0e"
GOOD = "#2ca02c"
BAD = "#d62728"


def _money_layout(fig: go.Figure, title: str, y_title: str = "$") -> go.Figure:
    fig.update_layout(title=title, hovermode="x unified", yaxis_title=y_title, xaxis_title="Years from purchase",
                      legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
                      margin=dict(t=50, l=10, r=10, b=10))
    fig.update_yaxes(tickformat="$,.0f")
    return fig


def net_worth_chart(r: Result, real: bool) -> go.Figure:
    t = np.arange(r.months + 1) / 12
    k = 1 / r.deflator if real else np.ones_like(r.deflator)
    og, on, rn = r.owner_gross[0] * k, r.owner_net[0] * k, r.renter[0] * k
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=og, name="Owner – equity + portfolio (keep home)", line=dict(color=OWNER)))
    fig.add_trace(go.Scatter(x=t, y=on, name="Owner – if sold (after realtor & legal)",
                             line=dict(color=OWNER_NET, dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=rn, name="Renter – portfolio", line=dict(color=RENTER)))
    for gap, base, label in ((r.gap_net[0], on, "if sold"), (r.gap_gross[0], og, "keep home")):
        for c in find_crossings(gap):
            m = c.month
            y = np.interp(m, np.arange(r.months + 1), base)
            winner = "owning" if c.direction == "owner" else "renting"
            fig.add_trace(go.Scatter(
                x=[m / 12], y=[y], mode="markers", showlegend=False,
                marker=dict(size=12, symbol="x", color=GOOD if c.direction == "owner" else BAD),
                hovertemplate=f"Crossover ({label}): {winner} takes the lead at year {c.year:.1f}<extra></extra>",
            ))
    return _money_layout(fig, "Net worth: owning vs renting" + (" (today's dollars)" if real else ""))


def sell_year_chart(tbl: pd.DataFrame, real: bool) -> go.Figure:
    k = 1 / tbl["deflator"] if real else 1
    gap_net = tbl["gap_net"] * k
    gap_gross = tbl["gap_gross"] * k
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tbl["year"], y=gap_net, name="Owner advantage if sold that year",
                         marker_color=np.where(gap_net >= 0, GOOD, BAD)))
    fig.add_trace(go.Scatter(x=tbl["year"], y=gap_gross, name="Owner advantage if kept (no selling costs)",
                             line=dict(color=OWNER)))
    fig.add_trace(go.Scatter(x=tbl["year"], y=-(tbl["selling_costs"] * k), name="Selling-cost drag",
                             line=dict(color="grey", dash="dot")))
    losing = tbl.loc[gap_net < 0, "year"]
    for y in losing:
        fig.add_vrect(x0=y - 0.5, x1=y + 0.5, fillcolor=BAD, opacity=0.07, line_width=0)
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig = _money_layout(fig, "If you sell in year k: owner net worth minus renter net worth", "Owner − renter")
    fig.update_xaxes(title="Sell at end of year")
    return fig


def sensitivity_heatmap(grid: pd.DataFrame, x_col: str, x_label: str, horizon: int,
                        current: tuple[float, float]) -> go.Figure:
    piv = grid.pivot(index="home_appreciation", columns=x_col, values="breakeven_years").sort_index()
    z = piv.values
    text = np.where(np.isnan(z), "never", np.char.mod("%.1f y", np.nan_to_num(z)))
    zplot = np.where(np.isnan(z), horizon + 5, z)
    xs, ys = piv.columns.values * 100, piv.index.values * 100
    fig = go.Figure(go.Heatmap(
        z=zplot, x=xs, y=ys, text=text, texttemplate="%{text}", colorscale="RdYlGn_r", zmin=0, zmax=horizon + 5,
        colorbar=dict(title="Years"), hovertemplate=f"Home appreciation %{{y:.1f}}%<br>{x_label} %{{x:.1f}}%<br>"
                                                     "Minimum holding period: %{text}<extra></extra>",
    ))
    fig.update_layout(title="Minimum holding period for owning to win", xaxis_title=x_label,
                      yaxis_title="Home appreciation / yr", margin=dict(t=50, l=10, r=10, b=10))
    fig.update_xaxes(tickvals=xs, ticktext=[f"{v:.1f}%" for v in xs])
    fig.update_yaxes(tickvals=ys, ticktext=[f"{v:.1f}%" for v in ys])
    fig.add_trace(go.Scatter(x=[current[1] * 100], y=[current[0] * 100], mode="markers", showlegend=False,
                             marker=dict(symbol="circle-open", size=34, color="black", line=dict(width=3)),
                             hoverinfo="skip"))
    return fig


def yearly_costs_chart(r: Result, real: bool) -> go.Figure:
    years = np.arange(r.months) // 12
    k = 1 / r.deflator[:-1] if real else 1
    fig = go.Figure()
    for name, arr in r.owner_costs.items():
        y = pd.Series(arr[0] * k).groupby(years).sum() / 12
        if y.abs().sum() > 0:
            fig.add_trace(go.Bar(x=y.index + 1, y=y.values, name=name))
    renter = pd.Series(r.renter_monthly_total * k).groupby(years).sum() / 12
    fig.add_trace(go.Scatter(x=renter.index + 1, y=renter.values, name="Renter: rent + tenant insurance",
                             line=dict(color=RENTER, width=3)))
    fig.update_layout(barmode="stack")
    fig = _money_layout(fig, "Monthly housing cost by year", "$ / month")
    fig.update_xaxes(title="Year")
    return fig


def unrecoverable_chart(df: pd.DataFrame, r: Result, real: bool) -> go.Figure:
    k = 1 / r.deflator[1:] if real else 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["month"] / 12, y=df["owner"] * k, name="Owner: interest, tax, fees, "
                             "insurance, upkeep, closing costs", line=dict(color=OWNER)))
    fig.add_trace(go.Scatter(x=df["month"] / 12, y=df["renter"] * k, name="Renter: rent + tenant insurance",
                             line=dict(color=RENTER)))
    return _money_layout(fig, "Cumulative unrecoverable costs (money that builds no wealth)")


def fan_chart(pct: pd.DataFrame, months: int, title: str, real_deflator: np.ndarray | None) -> go.Figure:
    t = np.arange(months + 1) / 12
    k = 1 / real_deflator if real_deflator is not None else 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=pct["p90"] * k, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=pct["p10"] * k, fill="tonexty", fillcolor="rgba(31,119,180,0.2)",
                             line=dict(width=0), name="P10–P90 range"))
    fig.add_trace(go.Scatter(x=t, y=pct["p50"] * k, name="Median", line=dict(color=OWNER)))
    fig.add_hline(y=0, line_color="black", line_width=1)
    return _money_layout(fig, title, "Owner − renter")


def probability_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["year"], y=df["p_owner_ahead_if_sold"], name="If sold that year",
                             line=dict(color=OWNER_NET, dash="dash")))
    fig.add_trace(go.Scatter(x=df["year"], y=df["p_owner_ahead_gross"], name="If home kept",
                             line=dict(color=OWNER)))
    fig.add_hline(y=0.5, line_color="grey", line_dash="dot")
    fig.update_layout(title="Probability owning is ahead", yaxis_tickformat=".0%", yaxis_range=[0, 1],
                      xaxis_title="Year", hovermode="x unified", margin=dict(t=60, l=10, r=10, b=10))
    return fig


def breakeven_histogram(months: np.ndarray, horizon: int) -> go.Figure:
    yrs = months[~np.isnan(months)] / 12
    fig = go.Figure(go.Histogram(x=yrs, xbins=dict(start=0, end=horizon + 1, size=1), marker_color=OWNER))
    fig.update_layout(title="Distribution of minimum holding period (simulations where owning wins)",
                      xaxis_title="Years", yaxis_title="Simulations", margin=dict(t=60, l=10, r=10, b=10))
    return fig


def line_chart(df: pd.DataFrame, title: str, y_title: str, fmt: str | None = None) -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=str(col), mode="lines"))
    fig.update_layout(title=title, yaxis_title=y_title, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0),
                      margin=dict(t=50, l=10, r=10, b=10))
    if fmt:
        fig.update_yaxes(tickformat=fmt)
    return fig
