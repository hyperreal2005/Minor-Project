"""Plotly building blocks with one visual system: recessive grid, ink-coloured text, a surface gap
between heatmap cells, hover on every mark, method identity by colour and symbol together."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import data as D

FONT = "Inter, Segoe UI, Arial, sans-serif"


def style(fig: go.Figure, *, height: int | None = None, legend: bool = True,
          margin: dict | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white", font=dict(family=FONT, size=13, color=D.INK),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
        margin=margin or dict(l=10, r=10, t=30, b=10),
        hoverlabel=dict(bgcolor="white", bordercolor=D.AXIS, font=dict(family=FONT, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=D.GRID, linecolor=D.AXIS, zerolinecolor=D.AXIS,
                     tickfont=dict(color=D.INK_2), title_font=dict(color=D.INK_2))
    fig.update_yaxes(gridcolor=D.GRID, linecolor=D.AXIS, zerolinecolor=D.AXIS,
                     tickfont=dict(color=D.INK_2), title_font=dict(color=D.INK_2))
    if height:
        fig.update_layout(height=height)
    return fig


def seq_scale():
    n = len(D.SEQUENTIAL) - 1
    return [[i / n, c] for i, c in enumerate(D.SEQUENTIAL)]


def div_scale(neg=D.RED, pos=D.BLUE):
    return [[0, neg], [0.5, D.NEUTRAL], [1, pos]]


def heatmap(z, x, y, *, text=None, hover=None, scale="seq", zmin=0, zmax=1, colorbar=None,
            height=None, bold=None) -> go.Figure:
    """Cells with a surface gap and luminance-chosen text; NaN cells read 'marked'."""
    z = np.asarray(z, dtype=float)
    cs = seq_scale() if scale == "seq" else div_scale() if scale == "div" else scale
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y, colorscale=cs, zmin=zmin, zmax=zmax, xgap=3, ygap=3,
        hovertext=hover, hoverinfo="text" if hover is not None else "x+y+z",
        colorbar=dict(title=dict(text=colorbar or "", side="right"), thickness=12,
                      outlinewidth=0, tickfont=dict(size=11)),
    ))
    # Cell labels as annotations, so ink can follow the cell's own luminance.
    if text is not None:
        from plotly.colors import sample_colorscale, unlabel_rgb

        for i, row in enumerate(text):
            for j, t in enumerate(row):
                v = z[i, j]
                if v != v:
                    t, color = (t or "marked"), D.MUTED
                else:
                    frac = float(np.clip((v - zmin) / (zmax - zmin), 0, 1)) if zmax > zmin else 0.0
                    rgb = sample_colorscale(cs, [frac])[0]
                    r, g, b = (c / 255 for c in unlabel_rgb(rgb)) if rgb.startswith("rgb") else (
                        int(rgb[k:k + 2], 16) / 255 for k in (1, 3, 5))
                    color = "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else D.INK
                b = bold is not None and bold[i][j]
                fig.add_annotation(x=x[j], y=y[i], text=f"<b>{t}</b>" if b else t,
                                   showarrow=False, font=dict(size=12, color=color))
    style(fig, height=height, legend=False)
    fig.update_xaxes(side="top", showgrid=False, linecolor="rgba(0,0,0,0)")
    fig.update_yaxes(autorange="reversed", showgrid=False, linecolor="rgba(0,0,0,0)")
    return fig


def method_trace(df, x, y, method, *, name=None, hover=None, size=10, opacity=1.0,
                 mode="markers", showlegend=True, **kw) -> go.Scatter:
    return go.Scatter(
        x=df[x], y=df[y], mode=mode, name=name or D.METHOD_LABEL.get(method, method),
        marker=dict(color=D.METHOD_COLOR.get(method, D.MUTED), size=size,
                    symbol=D.METHOD_SYMBOL.get(method, "circle"),
                    line=dict(color="white", width=1)),
        opacity=opacity, hovertext=hover, hoverinfo="text" if hover is not None else None,
        showlegend=showlegend, legendgroup=method, **kw)


def band_shape(fig, lo, hi, *, row=None, col=None, axis="x", label=True):
    """The retrains' band as a shaded strip across the plot."""
    if not (lo == lo and hi == hi):
        return
    # Outlined, so a band narrower than a pixel still shows where the retrains are.
    kw = dict(fillcolor=D.BAND_FILL, line=dict(color=D.RETRAIN, width=1), layer="below")
    if row is not None:
        kw.update(row=row, col=col)
    if axis == "x":
        fig.add_vrect(x0=lo, x1=hi, **kw)
    else:
        fig.add_hrect(y0=lo, y1=hi, **kw)


def pct(k, n) -> str:
    return f"{int(k)}/{int(n)}" if n == n and n else "n/a"
