"""Motion layer for the console, following .claude/skills/ui-animation/SKILL.md.

Rule that shapes everything here: Streamlit reruns the script on every interaction and re-renders changed
elements, so motion fires only when a value changed since the previous run (`changed`). Markup always carries
the final value, so the page is complete at rest even if the runtime script never runs.
"""
from __future__ import annotations

import html

import streamlit as st

# Timing constants (skill: easing defaults table + named curves)
EASE_ENTER = "cubic-bezier(0.22, 1, 0.36, 1)"
EASE_MOVE = "cubic-bezier(0.25, 1, 0.5, 1)"
EASE_PRESS = "cubic-bezier(0.25, 0.46, 0.45, 0.94)"
ROLL_MS = 420      # number arriving / driven by a control (expo-out)
MOVE_MS = 320      # meter fill travelling to the new score
SWAP_MS = 220      # verdict text/state swap
PRESS_MS = 160     # button release
CHART_MS = 300     # Plotly data tween

_STORE = "_motion"
_MISSING = object()


def changed(key: str, value):
    """Remember `value` under `key`; return the previous value only if it differs (else None)."""
    store = st.session_state.setdefault(_STORE, {})
    prev = store.get(key, _MISSING)
    store[key] = value
    if prev is _MISSING or prev == value:
        return None
    return prev


def roll(key: str, value: float, text: str, *, digits: int = 0, scale: float = 1.0,
         prefix: str = "", suffix: str = "", locale: str = "vi-VN") -> str:
    """A number that rolls from its previous value to `value` (text = final formatted value).
    `locale` must match how `text` is formatted (vi-VN for figures, en-US for code-style thresholds)."""
    prev = changed(key, round(float(value), 10))
    safe = html.escape(text)
    if prev is None:
        return f'<span class="cx-num">{safe}</span>'
    return (f'<span class="cx-num" data-roll-key="{html.escape(key)}" data-roll-from="{prev}" '
            f'data-roll-to="{float(value)}" data-roll-digits="{digits}" data-roll-scale="{scale}" '
            f'data-roll-locale="{html.escape(locale)}" data-roll-prefix="{html.escape(prefix)}" '
            f'data-roll-suffix="{html.escape(suffix)}">{safe}</span>')


def scale_attrs(key: str | None, fraction: float, style: str = "") -> str:
    """Attributes for an element whose transform: scaleX tweens from the previous fraction (key=None: static)."""
    fraction = max(0.0, min(1.0, float(fraction)))
    base = f'style="transform:scaleX({fraction:.4f});{html.escape(style)}"'
    if key is None:
        return base
    prev = changed(key, round(fraction, 6))
    if prev is None:
        return base
    return f'{base} data-scale-key="{html.escape(key)}" data-scale-from="{prev}" data-scale-to="{fraction}"'


def swap_class(key: str, state: str) -> str:
    """'cx-swap' when a discrete state (e.g. the verdict) changed since last run."""
    return "cx-swap" if changed(key, state) is not None else ""


def chart_transition() -> dict:
    return {"duration": CHART_MS, "easing": "cubic-out"}


CSS = f"""<style>
:root {{ --ease-enter:{EASE_ENTER}; --ease-move:{EASE_MOVE}; --ease-press:{EASE_PRESS}; }}

/* Press feedback: 0 ms down, eased release (skill: tappable controls press on :active at 0 ms). */
.stButton button, .stDownloadButton button, .stFormSubmitButton button,
[data-testid="stButtonGroup"] button {{
  touch-action: manipulation;
  transition: transform {PRESS_MS}ms var(--ease-press);
}}
.stButton button:active, .stDownloadButton button:active, .stFormSubmitButton button:active,
[data-testid="stButtonGroup"] button:active {{ transform: scale(0.97); transition-duration: 0ms; }}

/* State swap: blur(2px) hides the rough crossfade between the old and new verdict. */
@keyframes cx-swap {{
  from {{ opacity: 0; filter: blur(2px); transform: translateY(4px); }}
  to   {{ opacity: 1; filter: blur(0);   transform: none; }}
}}
.cx-swap {{ animation: cx-swap {SWAP_MS}ms var(--ease-enter) both; }}

/* Confirmation beside the control that caused it. */
@keyframes cx-confirm {{ from {{ opacity: 0; transform: translateX(-6px); }} to {{ opacity: 1; transform: none; }} }}
.cx-confirm {{ animation: cx-confirm {SWAP_MS}ms var(--ease-enter) both; }}

.cx-num.cx-rolling {{ font-variant-numeric: tabular-nums; }}
[data-testid="stElementContainer"]:has(.cx-runtime), .stElementContainer:has(.cx-runtime) {{ display: none; }}
.cx-meter-fill {{ transform-origin: left center; }}

@media (prefers-reduced-motion: reduce) {{
  /* Keep feedback, drop travel: crossfades stay, translate/scale/blur go. */
  @keyframes cx-swap {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
  @keyframes cx-confirm {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
  .stButton button:active, .stDownloadButton button:active, .stFormSubmitButton button:active,
  [data-testid="stButtonGroup"] button:active {{ transform: none; }}
}}
</style>"""

RUNTIME = f"""<script>
(() => {{
  if (window.__cxMotion) {{ window.__cxMotion.schedule(); return; }}
  const ROLL_MS = {ROLL_MS}, MOVE_MS = {MOVE_MS};
  const reduce = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const expoOut = t => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t));          // ~cubic-bezier(0.19,1,0.22,1)
  const moveOut = t => 1 - Math.pow(1 - t, 4);                            // ~cubic-bezier(0.25,1,0.5,1)
  const live = {{}};   // key -> {{ value, raf, running }} so a new target retargets from where the number is

  const fmt = (el, v) => {{
    const d = +el.dataset.rollDigits || 0, s = +el.dataset.rollScale || 1;
    const n = new Intl.NumberFormat(el.dataset.rollLocale || 'vi-VN',
      {{ minimumFractionDigits: d, maximumFractionDigits: d, useGrouping: true }}).format(v * s);
    return (el.dataset.rollPrefix || '') + n + (el.dataset.rollSuffix || '');
  }};

  function run(key, from, to, dur, ease, paint, done) {{
    const prev = live[key];
    if (prev && prev.running) {{ cancelAnimationFrame(prev.raf); from = prev.value; }}
    const state = {{ value: from, running: true, raf: 0 }};
    live[key] = state;
    if (reduce() || !isFinite(from) || from === to) {{ state.running = false; state.value = to; done(); return; }}
    const start = performance.now();
    paint(from);
    const step = now => {{
      const t = Math.min(1, (now - start) / dur);
      state.value = from + (to - from) * ease(t);
      if (t < 1) {{ paint(state.value); state.raf = requestAnimationFrame(step); }}
      else {{ state.running = false; state.value = to; done(); }}
    }};
    state.raf = requestAnimationFrame(step);
  }}

  function rollEl(el) {{
    el.dataset.rolled = '1';
    const finalText = el.textContent;
    el.classList.add('cx-rolling');
    run('n:' + el.dataset.rollKey, +el.dataset.rollFrom, +el.dataset.rollTo, ROLL_MS, expoOut,
        v => {{ el.textContent = fmt(el, v); }},
        () => {{ el.textContent = finalText; el.classList.remove('cx-rolling'); }});
  }}

  function scaleEl(el) {{
    el.dataset.rolled = '1';
    const to = +el.dataset.scaleTo;
    run('s:' + el.dataset.scaleKey, +el.dataset.scaleFrom, to, MOVE_MS, moveOut,
        v => {{ el.style.transform = 'scaleX(' + v + ')'; }},
        () => {{ el.style.transform = 'scaleX(' + to + ')'; }});
  }}

  function scan() {{
    document.querySelectorAll('[data-roll-to]:not([data-rolled])').forEach(rollEl);
    document.querySelectorAll('[data-scale-to]:not([data-rolled])').forEach(scaleEl);
  }}
  let queued = false;
  const schedule = () => {{ if (!queued) {{ queued = true; requestAnimationFrame(() => {{ queued = false; scan(); }}); }} }};
  new MutationObserver(schedule).observe(document.body, {{ childList: true, subtree: true }});
  window.__cxMotion = {{ schedule }};
  scan();
}})();
</script>"""


def inject() -> None:
    st.html(CSS)
    # Main column, not the sidebar (any sidebar content would bring the sidebar back with top navigation).
    # The marker lets CSS collapse the empty element so it adds no gap; the script has already run.
    st.html('<span class="cx-runtime" hidden></span>' + RUNTIME, unsafe_allow_javascript=True)
