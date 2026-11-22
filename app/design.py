"""Design system for the console (ui-design skill, product-UI track).

Visual thesis: a calm risk-review desk. Warm neutral canvas, white work surfaces lifted with a layered shadow,
one blue accent, status colours only for state, a translucent top bar for navigation.
Depth strategy: layered shadow in light mode, a 1px ring only in dark mode. One radius system: 12px surfaces,
6px controls, full pills. Spacing rhythm: 8 / 16 / 32px.
"""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

import motion

TOKENS = {
    "light": {
        "canvas": "#f3f2ee", "surface": "#ffffff", "raised": "#ffffff",
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#6f6e69",
        "grid": "#e8e7e1", "axis": "#c3c2b7", "divider": "rgba(11,11,11,0.08)",
        "accent": "#2a78d6", "accent_track": "#cde2fb", "accent_ink": "#1c5cab",
        "series_red": "#e34948", "deemph": "#c3c2b7",
        "good_text": "#006300", "bad_text": "#b52f2f",
        "shadow": "0 0 0 1px rgba(11,11,11,0.06), 0 1px 2px -1px rgba(11,11,11,0.06), 0 2px 4px 0 rgba(11,11,11,0.04)",
        "bar": "rgba(243,242,238,0.72)", "bar_edge": "rgba(11,11,11,0.06)",
    },
    "dark": {
        "canvas": "#0d0d0d", "surface": "#1a1a19", "raised": "#222220",
        "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#9a9993",
        "grid": "#2c2c2a", "axis": "#383835", "divider": "rgba(255,255,255,0.08)",
        "accent": "#3987e5", "accent_track": "#184f95", "accent_ink": "#86b6ef",
        "series_red": "#e66767", "deemph": "#52514e",
        "good_text": "#0ca30c", "bad_text": "#e66767",
        "shadow": "inset 0 0 0 1px rgba(255,255,255,0.06)",
        "bar": "rgba(13,13,13,0.72)", "bar_edge": "rgba(255,255,255,0.06)",
    },
}
# Status palette is fixed across themes; the label inside a fill is white, except on warning/serious.
STATUS = {
    "good": ("#0ca30c", "#ffffff"),
    "warning": ("#fab219", "#0b0b0b"),
    "serious": ("#ec835a", "#0b0b0b"),
    "critical": ("#d03b3b", "#ffffff"),
    "neutral": ("#898781", "#ffffff"),
}
RISK_TONE = {"Rất cao": "critical", "Cao": "serious", "Trung bình": "warning", "Thấp": "good"}
DRIFT_TONE = {"Trôi dạt": "critical", "Cần theo dõi": "warning", "Ổn định": "good"}
FONT = "IBM Plex Sans, Segoe UI, system-ui, sans-serif"
MONO = "IBM Plex Mono, Consolas, monospace"


def mode() -> str:
    try:
        return "dark" if (st.context.theme.get("type") or "light") == "dark" else "light"
    except Exception:  # bare mode / tests
        return "light"


def t() -> dict:
    return TOKENS[mode()]


def esc(text) -> str:
    return html.escape(str(text))


def inject_css() -> None:
    k = t()
    st.html(f"""<style>
:root {{
  --c-canvas:{k['canvas']}; --c-surface:{k['surface']}; --c-raised:{k['raised']};
  --c-ink:{k['ink']}; --c-ink2:{k['ink2']}; --c-muted:{k['muted']};
  --c-grid:{k['grid']}; --c-divider:{k['divider']}; --c-accent:{k['accent']};
  --c-track:{k['accent_track']}; --c-accent-ink:{k['accent_ink']};
  --c-good-text:{k['good_text']}; --c-bad-text:{k['bad_text']};
  --shadow-surface:{k['shadow']};
  --f-mono:{MONO};
}}

/* ---- Chrome: translucent top bar (materials guideline), quiet and out of the way ---- */
[data-testid="stDecoration"], [data-testid="stMainMenu"], [data-testid="stStatusWidget"] {{ display:none; }}
header[data-testid="stHeader"] {{
  background:{k['bar']};
  -webkit-backdrop-filter: blur(16px) saturate(180%); backdrop-filter: blur(16px) saturate(180%);
  box-shadow: 0 1px 0 {k['bar_edge']};
}}
@media (prefers-reduced-transparency: reduce) {{
  header[data-testid="stHeader"] {{ background: var(--c-canvas); -webkit-backdrop-filter:none; backdrop-filter:none; }}
}}
.block-container {{ padding-top: 5.25rem; padding-bottom: 4rem; max-width: 1280px; }}h1, h2, h3 {{ letter-spacing: -0.01em; text-wrap: balance; }}
code {{ font-family: var(--f-mono); }}
[data-testid="stFileUploaderDropzone"] svg,
[data-testid="stFileUploaderDropzoneInstructions"] [data-testid="stIconMaterial"] {{ display:none; }}

/* ---- Work surfaces: white, layered shadow, 12px radius; the page canvas stays quiet ---- */
[class*="st-key-surface"] {{
  background: var(--c-surface); border-radius: 12px; padding: 20px 24px;
  box-shadow: var(--shadow-surface);
}}

/* ---- Hero band: the page's orientation region in deep blue-violet (hues within ~25deg so it reads as
   one surface), lit from the top-left like every other surface. Data panels below stay light. ---- */
[class*="st-key-band"] {{
  position: relative; overflow: hidden; border-radius: 16px; padding: 28px 32px;
  background:
    radial-gradient(90% 120% at 0% 0%, rgba(120,170,255,0.28) 0%, rgba(120,170,255,0) 60%),
    radial-gradient(70% 90% at 100% 100%, rgba(139,92,246,0.30) 0%, rgba(139,92,246,0) 65%),
    linear-gradient(135deg, #0d2253 0%, #183a8a 52%, #2f2d86 100%);
  box-shadow: 0 1px 2px rgba(13,34,83,0.25), 0 18px 40px -18px rgba(24,58,138,0.55);
  color: #eef2ff;
}}
[class*="st-key-band"] .cx-title {{ color:#ffffff; font-size:34px; }}
[class*="st-key-band"] .cx-lede {{ color:rgba(238,242,255,0.82); }}
[class*="st-key-band"] .cx-chip {{ background:rgba(255,255,255,0.10); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.18);
                                   color:rgba(238,242,255,0.86); }}
[class*="st-key-band"] .cx-chip b {{ color:#ffffff; }}
[class*="st-key-band"] .cx-hero-value, [class*="st-key-band"] .cx-stat-value {{ color:#ffffff; }}
[class*="st-key-band"] .cx-hero-label, [class*="st-key-band"] .cx-stat-label {{ color:rgba(238,242,255,0.80); }}
[class*="st-key-band"] .cx-stat-note {{ color:rgba(238,242,255,0.62); }}
[class*="st-key-band"] .cx-stat:not(.row-start) {{ border-left-color: rgba(255,255,255,0.16); }}
[class*="st-key-band"] .cx-legend {{ color:rgba(238,242,255,0.80); }}
[class*="st-key-band"] [data-testid="stCaptionContainer"], [class*="st-key-band"] [data-testid="stCaptionContainer"] p,
[class*="st-key-band"] small {{ color:rgba(238,242,255,0.74); }}
[class*="st-key-band"] iframe {{ background: transparent; color-scheme: normal; }}
[class*="st-key-band"] .stButton button {{
  background: rgba(255,255,255,0.10); color:#ffffff; border: 1px solid rgba(255,255,255,0.22);
  transition: transform 160ms var(--ease-press), background-color 200ms ease;
}}
@media (hover: hover) and (pointer: fine) {{
  [class*="st-key-band"] .stButton button:hover {{ background: rgba(255,255,255,0.18); color:#ffffff; border-color: rgba(255,255,255,0.32); }}
}}
[class*="st-key-band"] .stButton button[kind="tertiary"] {{ background: transparent; border-color: transparent; color: rgba(238,242,255,0.82); }}
@media (max-width: 720px) {{ [class*="st-key-band"] {{ padding: 20px; }} [class*="st-key-band"] .cx-title {{ font-size:28px; }} }}

/* ---- Page header ---- */
.cx-header {{ display:flex; flex-direction:column; gap:8px; margin-bottom: 8px; }}
.cx-title {{ font-size:30px; line-height:1.15; font-weight:600; color:var(--c-ink); margin:0; }}
.cx-lede {{ font-size:15px; color:var(--c-ink2); max-width:76ch; line-height:1.55; margin:0; }}
.cx-chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:4px; }}
.cx-chip {{ font-size:12.5px; color:var(--c-ink2); background:var(--c-surface); box-shadow:var(--shadow-surface);
           border-radius:999px; padding:4px 12px; white-space:nowrap; }}
.cx-chip b {{ color:var(--c-ink); font-weight:600; }}
.cx-mono {{ font-family:var(--f-mono); font-variant-numeric: tabular-nums; }}

.cx-section {{ display:flex; flex-direction:column; gap:2px; margin: 0 0 4px; }}
.cx-section h3 {{ font-size:16px; font-weight:600; margin:0; color:var(--c-ink); }}
.cx-section p {{ font-size:13.5px; color:var(--c-ink2); margin:0; max-width:80ch; }}

/* ---- Stat row: dividers between siblings instead of a card per number ---- */
.cx-stats {{ display:grid; gap:16px 0; }}
.cx-stat {{ display:flex; flex-direction:column; gap:4px; min-width:0; padding:2px 20px; }}
.cx-stat.row-start {{ padding-left:0; }}
.cx-stat.row-end {{ padding-right:0; }}
.cx-stat:not(.row-start) {{ border-left:1px solid var(--c-divider); }}
.cx-stat-label {{ font-size:13px; color:var(--c-ink2); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.cx-stat-value {{ font-size:26px; font-weight:600; color:var(--c-ink); line-height:1.15; }}
.cx-stat-note {{ font-size:12.5px; color:var(--c-muted); line-height:1.4; }}
.cx-stat-note.up-good, .cx-stat-note.down-good {{ color:var(--c-good-text); }}
.cx-stat-note.up-bad, .cx-stat-note.down-bad {{ color:var(--c-bad-text); }}
.cx-stat .cx-pill {{ align-self:flex-start; }}
@media (max-width: 720px) {{
  .cx-stats {{ grid-template-columns: 1fr !important; }}
  .cx-stat, .cx-stat.row-start, .cx-stat.row-end {{ padding:12px 0 0; border-left:0; }}
  .cx-stat + .cx-stat {{ border-top:1px solid var(--c-divider); }}
}}

.cx-hero {{ display:flex; flex-direction:column; gap:6px; }}
.cx-hero-value {{ font-size:64px; font-weight:600; line-height:1; color:var(--c-ink); letter-spacing:-0.025em; }}
.cx-hero-label {{ font-size:14.5px; color:var(--c-ink2); max-width:44ch; line-height:1.5; }}

.cx-pill {{ display:inline-block; font-size:12px; font-weight:600; letter-spacing:0.01em; border-radius:999px;
           padding:3px 10px; line-height:1.4; white-space:nowrap; }}

.cx-verdict {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px 12px; }}
.cx-verdict-title {{ font-size:16px; font-weight:600; color:var(--c-ink); }}
.cx-verdict-body {{ font-size:13.5px; color:var(--c-ink2); flex-basis:100%; line-height:1.5; }}
.cx-banner {{ padding:14px 16px; border-radius:8px; background:var(--c-surface); box-shadow:var(--shadow-surface); }}

.cx-meter {{ position:relative; margin: 16px 0 28px; }}
.cx-meter-track {{ height:8px; border-radius:999px; background:var(--c-track); overflow:hidden; }}
.cx-meter-fill {{ height:100%; width:100%; }}
.cx-meter-tick {{ position:absolute; top:-5px; width:2px; height:18px; background:var(--c-ink); border-radius:1px; }}
.cx-meter-ticklabel {{ position:absolute; top:16px; transform:translateX(-50%); font-size:11.5px; color:var(--c-ink2);
                      white-space:nowrap; font-family:var(--f-mono); }}
.cx-meter-scale {{ display:flex; justify-content:space-between; font-size:11.5px; color:var(--c-muted); margin-top:4px; }}

.cx-facts {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(130px,1fr)); gap:8px 16px; margin:4px 0 0; }}
.cx-fact dt {{ font-size:12.5px; color:var(--c-ink2); }}
.cx-fact dd {{ margin:0; font-size:17px; font-weight:600; color:var(--c-ink); }}

.cx-matrix {{ border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:14px; }}
.cx-matrix th, .cx-matrix td {{ padding:8px 14px; border-bottom:1px solid var(--c-divider); text-align:right; }}
.cx-matrix th {{ font-weight:500; color:var(--c-ink2); text-align:left; }}
.cx-matrix td.hit {{ font-weight:600; color:var(--c-ink); }}

.cx-legend {{ display:flex; flex-wrap:wrap; gap:6px 16px; font-size:12.5px; color:var(--c-ink2); margin: 2px 0 4px; }}
.cx-legend span {{ display:inline-flex; align-items:center; gap:6px; }}
.cx-swatch {{ width:10px; height:10px; border-radius:3px; display:inline-block; }}

.cx-notes {{ font-size:14px; color:var(--c-ink2); line-height:1.6; max-width:96ch; margin:0; padding-left:18px; }}
.cx-notes li + li {{ margin-top:6px; }}
</style>""")
    motion.inject()


# ---------- building blocks ----------

def band(name: str):
    """The coloured hero band that opens a presented page (title, context, headline numbers, 3D scene)."""
    return st.container(key=f"band_{name}")


def surface(name: str):
    """A lifted work surface (white, layered shadow). Use one or two per page, never one per block."""
    return st.container(key=f"surface_{name}")


def pill(text: str, tone: str = "neutral") -> str:
    bg, fg = STATUS[tone]
    return f'<span class="cx-pill" style="background:{bg};color:{fg}">{esc(text)}</span>'


def header(title: str, lede: str, chips: list[str] | None = None, eyebrow: str | None = None) -> None:
    """Page title, one line of utility copy, and context chips. `eyebrow` is accepted but not shown
    (ui-design: an eyebrow above every heading is slop)."""
    chips_html = "".join(f'<span class="cx-chip">{c}</span>' for c in (chips or []))
    st.html(f"""<div class="cx-header">
  <h1 class="cx-title">{esc(title)}</h1>
  <p class="cx-lede">{esc(lede)}</p>
  {f'<div class="cx-chips">{chips_html}</div>' if chips_html else ''}
</div>""")


def section(title: str, note: str | None = None) -> None:
    st.html(f'<div class="cx-section"><h3>{esc(title)}</h3>{f"<p>{esc(note)}</p>" if note else ""}</div>')


def _value_html(it: dict) -> str:
    """Stat/hero value: rolls from its previous value when the item has `key` and numeric `raw`."""
    if it.get("key") and it.get("raw") is not None:
        return motion.roll(it["key"], it["raw"], it["value"], digits=it.get("digits", 0),
                           scale=it.get("scale", 1.0), prefix=it.get("prefix", ""), suffix=it.get("suffix", ""))
    return esc(it["value"])


def tiles(items: list[dict], cols: int | None = None) -> None:
    """A divider-separated stat row. items: {label, value, note?, trend? ('up-good'|'up-bad'|'down-good'|
    'down-bad'), status?: (text, tone), key?/raw?/digits?/scale?/suffix?/prefix? for a value that rolls}."""
    cols = cols or len(items)
    out = []
    for i, it in enumerate(items):
        pos = (" row-start" if i % cols == 0 else "") + (" row-end" if i % cols == cols - 1 else "")
        note = f'<div class="cx-stat-note {it.get("trend", "")}">{esc(it["note"])}</div>' if it.get("note") else ""
        status = ""
        if it.get("status"):
            swap = motion.swap_class(f'{it["key"]}.status', it["status"][0]) if it.get("key") else ""
            status = f'<span class="{swap}">{pill(*it["status"])}</span>' if swap else pill(*it["status"])
        out.append(f'<div class="cx-stat{pos}"><div class="cx-stat-label" title="{esc(it["label"])}">'
                   f'{esc(it["label"])}</div><div class="cx-stat-value">{_value_html(it)}</div>{status}{note}</div>')
    st.html(f'<div class="cx-stats" style="grid-template-columns:repeat({cols}, minmax(0, 1fr))">{"".join(out)}</div>')


def hero(value: str, label: str, **roll) -> None:
    """roll: key=, raw=, digits=, scale=, suffix= to roll the figure when it changes."""
    st.html(f'<div class="cx-hero"><div class="cx-hero-value">{_value_html({"value": value, **roll})}</div>'
            f'<div class="cx-hero-label">{esc(label)}</div></div>')


def verdict(text: str, tone: str, title: str, body: str, swap_key: str | None = None, boxed: bool = False) -> None:
    """swap_key: crossfade the verdict when its decision text changed since the previous run."""
    swap = motion.swap_class(swap_key, text) if swap_key else ""
    title_html = f'<span class="cx-verdict-title">{esc(title)}</span>' if title else ""
    box = " cx-banner" if boxed else ""
    st.html(f'<div class="cx-verdict{box} {swap}">{pill(text, tone)}{title_html}'
            f'<span class="cx-verdict-body">{body}</span></div>')


def meter(value: float, threshold: float, scale_max: float, tone: str, key: str | None = None) -> None:
    """Probability meter: fill = score (severity colour), tick = threshold, track = lighter accent step.
    The fill is drawn with transform: scaleX (not width) so it can travel to a new score cheaply."""
    frac = min(value / scale_max, 1.0)
    th = min(threshold / scale_max, 1.0) * 100
    fill = STATUS[tone][0]
    attrs = motion.scale_attrs(key, frac, style=f"background:{fill}")
    st.html(f"""<div class="cx-meter" role="img" aria-label="Điểm {value:.3f}, ngưỡng {threshold:.4f}">
  <div class="cx-meter-track"><div class="cx-meter-fill" {attrs}></div></div>
  <div class="cx-meter-tick" style="left:calc({th:.2f}% - 1px)"></div>
  <div class="cx-meter-ticklabel" style="left:{th:.2f}%">ngưỡng {threshold:.4f}</div>
</div>
<div class="cx-meter-scale"><span>0%</span><span>{scale_max:.0%}</span></div>""")


def facts(items: list[tuple[str, str]]) -> None:
    rows = "".join(f'<div class="cx-fact"><dt>{esc(k)}</dt><dd>{v}</dd></div>' for k, v in items)
    st.html(f'<dl class="cx-facts">{rows}</dl>')


def legend(items: list[tuple[str, str]]) -> None:
    rows = "".join(f'<span><i class="cx-swatch" style="background:{c}"></i>{esc(n)}</span>' for n, c in items)
    st.html(f'<div class="cx-legend">{rows}</div>')


def banner(tone: str, label: str, body: str, swap_key: str | None = None) -> None:
    verdict(label, tone, "", body, swap_key=swap_key, boxed=True)


# ---------- charts ----------

def style(fig: go.Figure, height: int = 340, legend_on: bool = False, xtitle: str | None = None,
          ytitle: str | None = None, animate: bool = False) -> go.Figure:
    """animate=True tweens data between reruns. Use it only when the chart keeps the same categories and a
    stable `key` in show(); otherwise bars would travel between unrelated items."""
    if animate:
        fig.update_layout(transition=motion.chart_transition())
    k = t()
    axis = dict(gridcolor=k["grid"], gridwidth=1, linecolor=k["axis"], zeroline=False, showline=True,
                tickfont=dict(color=k["muted"], size=11.5), title_font=dict(color=k["ink2"], size=12.5),
                ticks="", automargin=True)
    fig.update_layout(
        height=height, margin=dict(l=8, r=16, t=16, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=k["ink2"], size=12.5),
        hoverlabel=dict(bgcolor=k["raised"], bordercolor=k["grid"], font=dict(color=k["ink"], family=FONT)),
        showlegend=legend_on,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=k["ink2"])),
        barcornerradius=4, bargap=0.35, separators=",.",
    )
    fig.update_xaxes(**axis, title_text=xtitle)
    fig.update_yaxes(**axis, title_text=ytitle)
    return fig


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False, "displaylogo": False}, key=key)
