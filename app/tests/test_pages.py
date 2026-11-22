"""Smoke tests: every page renders without an exception (requires artifacts from app/train.py)."""
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (APP / "artifacts" / "meta.json").exists(), reason="run app/train.py first")

PAGES = ["overview", "score_one", "batch", "policy", "drift"]
# Emoji / pictograph ranges - the UI must not display icons.
ICON_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿←-⇿⬀-⯿]")


def run(page: str) -> AppTest:
    """Start the real entrypoint (with st.navigation) and open the requested page."""
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=180)
    at.run()
    if page != "overview":
        at.switch_page(f"views/{page}.py").run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES + ["../Home"])
def test_no_icons_in_source(page):
    src = (APP / "views" / f"{page}.py").read_text(encoding="utf-8")
    assert not ICON_RE.search(src), ICON_RE.findall(src)
    assert "icon=" not in src


def test_batch_page_scores_sample():
    at = run("batch")
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["batch_flagged"] > 0


def test_group_policy_balances_fpr():
    at = run("policy")
    at.segmented_control(key="pol_mode").set_value("Theo nhóm tuổi").run()
    assert not at.exception, [e.value for e in at.exception]
    assert 0.8 <= at.session_state["policy_preview"]["Tỷ số FPR"] <= 1.25


def test_score_one_random_fraud_sample():
    at = run("score_one")
    at.button[1].click().run()  # "Hồ sơ gian lận tháng 7"
    assert not at.exception, [e.value for e in at.exception]
    assert 0.0 <= at.session_state["last_score"] <= 1.0


# ---------- motion (ui-animation skill rules) ----------

def _html(at: AppTest) -> str:
    return "\n".join(str(getattr(e.proto, "body", "")) for e in at.get("html"))


def test_motion_only_after_a_change():
    at = run("score_one")
    assert 'data-roll-to="' not in _html(at), "nothing may animate on first render"
    at.run()  # rerun with the same inputs: still no animation
    assert 'data-roll-to="' not in _html(at)
    at.button[1].click().run()  # load a fraud sample -> score changes
    assert not at.exception, [e.value for e in at.exception]
    body = _html(at)
    assert 'data-roll-key="score.p"' in body and 'data-scale-key="score.meter"' in body


def test_changed_helper():
    import sys
    sys.path.insert(0, str(APP))
    import streamlit as st
    import motion
    st.session_state.clear()
    assert motion.changed("k", 1) is None
    assert motion.changed("k", 1) is None
    assert motion.changed("k", 2) == 1


def test_motion_css_follows_skill():
    """Skill validation grep: no `transition: all`, no animated layout properties."""
    py = (APP / "motion.py").read_text(encoding="utf-8") + (APP / "design.py").read_text(encoding="utf-8")
    src = "\n".join(re.findall(r"<style>(.*?)</style>", py, flags=re.S))  # CSS only
    assert src and not re.search(r"transition\s*:\s*all", src)
    for prop in ("width", "height", "top", "left"):
        assert not re.search(rf"transition[^;{{}}]*\b{prop}\b", src), prop
    keyframes = re.findall(r"@keyframes[^{]+\{(.*?)\}\s*\}", src, flags=re.S)
    assert keyframes and not any(re.search(r"\b(width|height|top|left)\s*:", k) for k in keyframes)
