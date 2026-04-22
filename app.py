# app.py
# Streamlit MVP app for the Blackjack AI Coach
#
# This app connects:
# - blackjack_engine.py for the recommendation
# - optional nanoGPT model output for explanation text
#
# To run:
#   streamlit run app.py
#
# Optional files used:
# - blackjack_engine.py
# - model.py
# - out/best_model.pt
# - out/meta.json

import html
import json
from pathlib import Path

import streamlit as st
import torch

from blackjack_engine import format_for_gpt, recommend_action
from model import GPT, GPTConfig


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Blackjack AI Coach",
    page_icon="BJ",
    layout="wide",
)


# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg: #f7f0e6;
        --panel: rgba(255, 252, 247, 0.88);
        --panel-strong: rgba(255, 248, 238, 0.96);
        --ink: #1d2a22;
        --muted: #5f675f;
        --green: #1d6b4f;
        --green-deep: #113f30;
        --gold: #d7a94b;
        --red: #7f2f2a;
        --line: rgba(29, 42, 34, 0.12);
        --shadow: 0 18px 60px rgba(28, 45, 34, 0.10);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(215, 169, 75, 0.20), transparent 34%),
            radial-gradient(circle at top right, rgba(29, 107, 79, 0.22), transparent 28%),
            linear-gradient(180deg, #fbf6ef 0%, #f2e6d6 100%);
        color: var(--ink);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--ink);
        letter-spacing: -0.02em;
    }

    p, li, label, .stMarkdown, .stTextInput, .stSelectbox, .stRadio, .stCheckbox {
        font-family: 'Space Grotesk', sans-serif !important;
    }

    .stApp p,
    .stApp li,
    .stApp label,
    .stApp span,
    .stApp div,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stTextInputRootElement"] input,
    [data-baseweb="select"] span,
    div[data-testid="metric-container"] label,
    div[data-testid="metric-container"] [data-testid="stMetricLabel"],
    div[data-testid="metric-container"] [data-testid="stMetricValue"],
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }

    code, pre {
        font-family: 'IBM Plex Mono', monospace !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(20, 52, 39, 0.96), rgba(12, 35, 27, 0.98));
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f7f2ea !important;
    }

    [data-testid="stSidebar"] .stCodeBlock {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 14px;
    }

    .hero-shell {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(17, 63, 48, 0.10);
        border-radius: 28px;
        padding: 1.8rem 1.8rem 1.6rem 1.8rem;
        margin-bottom: 1.25rem;
        background:
            linear-gradient(140deg, rgba(18, 56, 42, 0.95), rgba(22, 83, 62, 0.92) 55%, rgba(215, 169, 75, 0.88) 145%);
        box-shadow: var(--shadow);
        color: #fff8ef;
    }

    .hero-shell,
    .hero-shell * {
        color: #fff8ef !important;
    }

    .hero-shell::after {
        content: "";
        position: absolute;
        inset: auto -8% -30% auto;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,0.22), transparent 68%);
        pointer-events: none;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-size: 0.72rem;
        opacity: 0.78;
        margin-bottom: 0.55rem;
        font-weight: 700;
    }

    .hero-title {
        font-size: clamp(2.2rem, 4vw, 4rem);
        line-height: 0.95;
        margin: 0 0 0.8rem 0;
        max-width: 760px;
    }

    .hero-copy {
        max-width: 760px;
        color: rgba(255, 248, 239, 0.88) !important;
        font-size: 1.04rem;
        line-height: 1.6;
        margin-bottom: 0;
    }

    .panel {
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        background: var(--panel);
        box-shadow: var(--shadow);
        backdrop-filter: blur(8px);
        margin-bottom: 1rem;
    }

    .panel-strong {
        background:
            linear-gradient(180deg, rgba(255, 248, 238, 0.97), rgba(245, 237, 226, 0.96));
    }

    .section-kicker {
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--green);
        margin-bottom: 0.45rem;
    }

    .section-title {
        font-size: 1.32rem;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 0.4rem;
    }

    .section-copy {
        color: var(--muted);
        line-height: 1.55;
        margin: 0;
    }

    .coach-call {
        border-radius: 22px;
        padding: 1.15rem 1.2rem;
        margin-top: 0.8rem;
        background:
            linear-gradient(135deg, rgba(17, 63, 48, 0.98), rgba(29, 107, 79, 0.95));
        color: #fbf6ef;
        border: 1px solid rgba(17, 63, 48, 0.10);
    }

    .coach-call-label {
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.7rem;
        opacity: 0.72;
        margin-bottom: 0.5rem;
    }

    .coach-call-text {
        font-size: 1.12rem;
        line-height: 1.6;
        margin: 0;
        white-space: pre-line;
    }

    .insight-card {
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1rem;
        min-height: 170px;
        background: rgba(255, 253, 249, 0.92);
        margin-bottom: 1rem;
    }

    .insight-title {
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--green);
        margin-bottom: 0.65rem;
    }

    .insight-copy {
        color: var(--ink);
        line-height: 1.6;
        margin: 0;
    }

    .chips-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 0.9rem;
    }

    .chip {
        border-radius: 999px;
        border: 1px solid rgba(17, 63, 48, 0.10);
        background: rgba(215, 169, 75, 0.14);
        color: var(--green-deep);
        padding: 0.45rem 0.85rem;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .tier-summary {
        border: 1px solid rgba(215, 169, 75, 0.28);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.8rem;
        background: linear-gradient(135deg, rgba(18, 56, 42, 0.97), rgba(29, 107, 79, 0.94));
        box-shadow: 0 10px 24px rgba(17, 63, 48, 0.16);
    }

    .tier-summary *,
    .tier-summary p {
        color: #fff8ef !important;
    }

    .tier-summary-kicker {
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.68rem;
        font-weight: 700;
        opacity: 0.75;
        margin-bottom: 0.45rem;
    }

    .tier-summary-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .tier-summary-price {
        color: #f2c96a !important;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }

    .tier-summary-copy {
        line-height: 1.5;
        margin: 0;
        opacity: 0.88;
    }

    .copy-block {
        display: block;
        white-space: pre-wrap;
        word-break: break-word;
        margin: 0.6rem 0 0.9rem 0;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(215, 169, 75, 0.35);
        background: linear-gradient(135deg, rgba(18, 56, 42, 0.98), rgba(29, 107, 79, 0.96));
        color: #fff8ef !important;
        -webkit-text-fill-color: #fff8ef !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.86rem;
        line-height: 1.55;
        box-shadow: 0 10px 24px rgba(17, 63, 48, 0.18);
    }

    .copy-block * {
        color: #fff8ef !important;
        -webkit-text-fill-color: #fff8ef !important;
    }

    div[data-testid="metric-container"] {
        background: rgba(255, 252, 247, 0.86);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1rem 1rem 0.9rem 1rem;
        box-shadow: 0 10px 30px rgba(28, 45, 34, 0.06);
    }

    div[data-testid="metric-container"] label {
        color: #000000 !important;
        font-weight: 600;
    }

    .stButton > button {
        width: 100%;
        border-radius: 18px;
        border: 0;
        background: linear-gradient(135deg, #173f31, #226b4f);
        color: #fff9f1;
        font-weight: 700;
        min-height: 3.2rem;
        box-shadow: 0 12px 30px rgba(17, 63, 48, 0.22);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #123427, #1d5b44);
    }

    .stTextInput > div > div,
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        border-radius: 16px !important;
    }

    .stSelectbox [data-baseweb="select"] > div {
        background: linear-gradient(135deg, rgba(18, 56, 42, 0.96), rgba(29, 107, 79, 0.94)) !important;
        border: 1px solid rgba(215, 169, 75, 0.35) !important;
        box-shadow: 0 10px 24px rgba(17, 63, 48, 0.18) !important;
    }

    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] svg {
        color: #fff8ef !important;
        fill: #fff8ef !important;
    }

    div[data-baseweb="popover"] div[role="listbox"] {
        background: linear-gradient(180deg, rgba(18, 56, 42, 0.98), rgba(29, 107, 79, 0.96)) !important;
        border: 1px solid rgba(215, 169, 75, 0.35) !important;
    }

    div[data-baseweb="popover"] div[role="option"] {
        background: transparent !important;
        color: #fff8ef !important;
    }

    div[data-baseweb="popover"] div[role="option"]:hover,
    div[data-baseweb="popover"] div[aria-selected="true"] {
        background: rgba(215, 169, 75, 0.22) !important;
        color: #ffffff !important;
    }

    [data-testid="stCodeBlock"],
    [data-testid="stCodeBlock"] pre,
    [data-testid="stCodeBlock"] code,
    [data-testid="stCodeBlock"] pre > div,
    [data-testid="stCodeBlock"] pre div,
    [data-testid="stCodeBlock"] code > span,
    [data-testid="stJson"],
    [data-testid="stJson"] pre,
    [data-testid="stJson"] code,
    [data-testid="stJson"] pre > div,
    [data-testid="stJson"] pre div,
    .stCodeBlock,
    .stCodeBlock pre,
    .stCodeBlock code,
    .stCodeBlock pre > div,
    .stCodeBlock pre div {
        background: linear-gradient(135deg, rgba(18, 56, 42, 0.98), rgba(29, 107, 79, 0.96)) !important;
        background-color: rgba(18, 56, 42, 0.98) !important;
        color: #fff8ef !important;
        border-color: rgba(215, 169, 75, 0.35) !important;
    }

    [data-testid="stCodeBlock"],
    [data-testid="stJson"],
    .stCodeBlock {
        border: 1px solid rgba(215, 169, 75, 0.35) !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 24px rgba(17, 63, 48, 0.18) !important;
        overflow: hidden !important;
    }

    [data-testid="stCodeBlock"] *,
    [data-testid="stJson"] *,
    [data-testid="stCodeBlock"] pre *,
    [data-testid="stCodeBlock"] code *,
    [data-testid="stCodeBlock"] [class*="token"],
    [data-testid="stCodeBlock"] [class*="language"],
    .stCodeBlock * {
        color: #fff8ef !important;
        -webkit-text-fill-color: #fff8ef !important;
        text-shadow: none !important;
    }

    [data-testid="stCodeBlock"] button,
    [data-testid="stJson"] button,
    .stCodeBlock button {
        background: rgba(255, 248, 239, 0.14) !important;
        border: 1px solid rgba(255, 248, 239, 0.24) !important;
        color: #fff8ef !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid rgba(215, 169, 75, 0.28) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        background: linear-gradient(135deg, rgba(18, 56, 42, 0.96), rgba(29, 107, 79, 0.94)) !important;
        box-shadow: 0 10px 24px rgba(17, 63, 48, 0.16) !important;
    }

    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] > details > summary {
        background: linear-gradient(135deg, rgba(18, 56, 42, 0.98), rgba(29, 107, 79, 0.95)) !important;
        color: #fff8ef !important;
    }

    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] svg {
        color: #fff8ef !important;
        fill: #fff8ef !important;
    }

    [data-testid="stExpanderDetails"] {
        background: linear-gradient(135deg, rgba(18, 56, 42, 0.94), rgba(29, 107, 79, 0.90)) !important;
        border-top: 1px solid rgba(215, 169, 75, 0.22) !important;
    }

    [data-testid="stExpanderDetails"],
    [data-testid="stExpanderDetails"] *,
    [data-testid="stExpanderDetails"] p,
    [data-testid="stExpanderDetails"] span,
    [data-testid="stExpanderDetails"] div,
    [data-testid="stExpanderDetails"] label,
    [data-testid="stExpanderDetails"] strong,
    [data-testid="stExpanderDetails"] code,
    [data-testid="stExpanderDetails"] pre,
    [data-testid="stExpanderDetails"] li,
    [data-testid="stExpanderDetails"] svg,
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"],
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stExpanderDetails"] [data-testid="stCodeBlock"] *,
    [data-testid="stExpanderDetails"] [data-testid="stJson"] * {
        color: #fff8ef !important;
        fill: #fff8ef !important;
        -webkit-text-fill-color: #fff8ef !important;
        text-shadow: none !important;
    }

    [data-testid="stExpanderDetails"] [data-testid="stCodeBlock"],
    [data-testid="stExpanderDetails"] [data-testid="stCodeBlock"] pre,
    [data-testid="stExpanderDetails"] [data-testid="stCodeBlock"] code,
    [data-testid="stExpanderDetails"] [data-testid="stCodeBlock"] pre > div,
    [data-testid="stExpanderDetails"] [data-testid="stCodeBlock"] pre div,
    [data-testid="stExpanderDetails"] [data-testid="stCodeBlock"] code > span,
    [data-testid="stExpanderDetails"] [data-testid="stJson"],
    [data-testid="stExpanderDetails"] [data-testid="stJson"] pre,
    [data-testid="stExpanderDetails"] [data-testid="stJson"] code,
    [data-testid="stExpanderDetails"] [data-testid="stJson"] pre > div,
    [data-testid="stExpanderDetails"] [data-testid="stJson"] pre div {
        background: linear-gradient(135deg, rgba(18, 56, 42, 0.98), rgba(29, 107, 79, 0.96)) !important;
        background-color: rgba(18, 56, 42, 0.98) !important;
    }

    .stRadio > div {
        gap: 0.35rem;
    }

    .footer-note {
        color: var(--muted);
        margin: 0;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
VALID_CARDS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

PLAN_TIERS = {
    "Table Coach - Free": {
        "kicker": "Starter",
        "name": "Table Coach",
        "price": "Free",
        "model": "Classical Coach Model",
        "level": "free",
        "copy": "Clean recommendations and plain-English hand lessons for practice sessions.",
    },
    "EV Edge - $19 / month": {
        "kicker": "Pro",
        "name": "EV Edge",
        "price": "$19 / month",
        "model": "Expected Value Model",
        "level": "pro",
        "copy": "Live expected-value comparisons for every legal move at the table.",
    },
    "Bankroll Desk - $39 / month": {
        "kicker": "Elite",
        "name": "Bankroll Desk",
        "price": "$39 / month",
        "model": "Bankroll-Aware EV Model",
        "level": "elite",
        "copy": "Decision support tuned for bet sizing, risk, and long-run discipline.",
    },
}

COACH_MODES = {
    "Table Coach": "beginner",
    "EV Edge": "math",
    "Bankroll Desk": "math",
}

PLAN_TO_MODE = {
    "Table Coach - Free": "Table Coach",
    "EV Edge - $19 / month": "EV Edge",
    "Bankroll Desk - $39 / month": "Bankroll Desk",
}

MODE_TO_PLAN = {mode: plan for plan, mode in PLAN_TO_MODE.items()}

if (
    "membership_tier" not in st.session_state
    or st.session_state.membership_tier not in PLAN_TIERS
):
    st.session_state.membership_tier = "Table Coach - Free"

if (
    "coach_mode" not in st.session_state
    or st.session_state.coach_mode not in COACH_MODES
):
    st.session_state.coach_mode = PLAN_TO_MODE[st.session_state.membership_tier]


def sync_coach_mode_from_plan():
    st.session_state.coach_mode = PLAN_TO_MODE[st.session_state.membership_tier]


def sync_plan_from_coach_mode():
    st.session_state.membership_tier = MODE_TO_PLAN[st.session_state.coach_mode]


def normalize_for_display(cards):
    return ",".join(cards)


def escape_text(text):
    return html.escape(str(text))


def selected_coach_text(coach_payload, explanation_mode):
    return coach_payload.get(COACH_MODES[explanation_mode], coach_payload.get("beginner", ""))


def render_panel(kicker, title, copy, strong=False):
    class_name = "panel panel-strong" if strong else "panel"
    st.markdown(
        f"""
        <section class="{class_name}">
            <div class="section-kicker">{escape_text(kicker)}</div>
            <div class="section-title">{escape_text(title)}</div>
            <p class="section-copy">{escape_text(copy)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_insight_card(title, body):
    st.markdown(
        f"""
        <section class="insight-card">
            <div class="insight-title">{escape_text(title)}</div>
            <p class="insight-copy">{escape_text(body)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_tier_summary(tier):
    st.markdown(
        f"""
        <section class="tier-summary">
            <div class="tier-summary-kicker">{escape_text(tier["kicker"])}</div>
            <div class="tier-summary-title">{escape_text(tier["name"])}</div>
            <div class="tier-summary-price">{escape_text(tier["price"])}</div>
            <div class="tier-summary-price">{escape_text(tier["model"])}</div>
            <p class="tier-summary-copy">{escape_text(tier["copy"])}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_copy_block(text):
    st.markdown(
        f"""<pre class="copy-block">{escape_text(text)}</pre>""",
        unsafe_allow_html=True,
    )


def render_json_block(payload):
    render_copy_block(json.dumps(payload, indent=2))


def bankroll_context(result, selected_tier, bankroll_amount=None, bet_size=None):
    action = result["recommended_action"]
    best_ev = result.get("best_ev")
    ev_margin = result.get("ev_margin")

    if selected_tier["level"] != "elite":
        return None

    if best_ev is None:
        return "This hand does not have enough EV detail for bankroll framing."

    if best_ev > 0.25:
        pressure = "high-value"
    elif best_ev > 0:
        pressure = "thin positive"
    elif best_ev > -0.25:
        pressure = "damage-control"
    else:
        pressure = "defensive"

    margin_text = (
        f"The model separates the top play by {ev_margin:+.3f} units, so the edge is fairly clear."
        if ev_margin is not None and ev_margin >= 0.08
        else "The model sees a narrow edge, so disciplined execution matters more than aggression."
    )

    if not bankroll_amount or bankroll_amount <= 0 or not bet_size or bet_size <= 0:
        return {
            "action": action,
            "best_ev": best_ev,
            "ev_margin": ev_margin,
            "pressure": pressure,
            "margin_text": margin_text,
            "has_bankroll_inputs": False,
            "bankroll_amount": bankroll_amount,
            "bet_size": bet_size,
            "bet_fraction": None,
            "stake_note": None,
            "conservative_unit": None,
            "assertive_unit": None,
            "expected_profit_per_hand": None,
            "expected_profit_per_100": None,
            "long_run_note": "Add a bankroll amount and live bet size to unlock the full bankroll analysis.",
        }

    bet_fraction = bet_size / bankroll_amount
    conservative_unit = bankroll_amount * 0.01
    assertive_unit = bankroll_amount * 0.02
    expected_profit_per_hand = best_ev * bet_size
    expected_profit_per_100 = expected_profit_per_hand * 100

    if bet_fraction > 0.03:
        stake_note = "very aggressive for a training bankroll"
    elif bet_fraction > 0.02:
        stake_note = "on the aggressive side"
    elif bet_fraction >= 0.01:
        stake_note = "within a disciplined working range"
    else:
        stake_note = "conservative"

    if best_ev > 0:
        long_run_note = (
            f"If you can repeatedly find spots with a positive edge and hold the same bankroll discipline, "
            f"this stake size implies about ${expected_profit_per_hand:,.2f} of expectation per hand, or roughly "
            f"${expected_profit_per_100:,.2f} per 100 similar hands. That is not a promise of short-term profit; "
            "it is the long-run value of making mathematically sound decisions without overbetting."
        )
    else:
        long_run_note = (
            f"This hand is not a positive-expectation betting opportunity at the current model output, so the bankroll goal "
            "is protection rather than growth. In negative or thin spots, long-run profit comes more from avoiding oversized bets "
            "and preserving capital for stronger situations than from pressing action."
        )

    return {
        "action": action,
        "best_ev": best_ev,
        "ev_margin": ev_margin,
        "pressure": pressure,
        "margin_text": margin_text,
        "has_bankroll_inputs": True,
        "bankroll_amount": bankroll_amount,
        "bet_size": bet_size,
        "bet_fraction": bet_fraction,
        "stake_note": stake_note,
        "conservative_unit": conservative_unit,
        "assertive_unit": assertive_unit,
        "expected_profit_per_hand": expected_profit_per_hand,
        "expected_profit_per_100": expected_profit_per_100,
        "long_run_note": long_run_note,
    }


def bankroll_coach_call(result, selected_tier, bankroll_amount=None, bet_size=None):
    ctx = bankroll_context(result, selected_tier, bankroll_amount, bet_size)
    if ctx is None:
        return None

    if not ctx["has_bankroll_inputs"]:
        return (
            f"{ctx['action']} is still the right move here, but Bankroll Desk gets much stronger once you enter bankroll and bet size. "
            f"Right now the hand grades as a {ctx['pressure']} spot at {ctx['best_ev']:+.3f} units."
        )

    return (
        f"{ctx['action']} is the bankroll-aware play here. At ${ctx['bet_size']:,.2f} on a ${ctx['bankroll_amount']:,.2f} bankroll, "
        f"you are risking {ctx['bet_fraction'] * 100:.2f}% of bankroll, so the edge only pays off cleanly if the stake stays disciplined. "
        f"This hand is worth {ctx['best_ev']:+.3f} units per original bet."
    )


def bankroll_lens_text(result, selected_tier, bankroll_amount=None, bet_size=None):
    ctx = bankroll_context(result, selected_tier, bankroll_amount, bet_size)
    if ctx is None:
        return None

    if not ctx["has_bankroll_inputs"]:
        return (
            f"This is a {ctx['pressure']} spot with the recommended action at {ctx['best_ev']:+.3f} units per original bet. "
            f"{ctx['margin_text']} Add bankroll and bet size to see whether the stake is conservative, balanced, or too aggressive. "
            "That extra layer is what turns a correct move into a usable bankroll plan."
        )

    return (
        f"With a bankroll of ${ctx['bankroll_amount']:,.2f} and a ${ctx['bet_size']:,.2f} stake, you are risking "
        f"{ctx['bet_fraction'] * 100:.2f}% of bankroll on this hand, which is {ctx['stake_note']}. "
        f"{ctx['margin_text']} A disciplined unit for this bankroll is roughly ${ctx['conservative_unit']:,.2f} to "
        f"${ctx['assertive_unit']:,.2f}, so this bet should be judged against that range. "
        f"If the unit size drifts too far above that range, even correct decisions become harder to monetize over time."
    )


def bankroll_guidance(result, selected_tier, bankroll_amount=None, bet_size=None):
    ctx = bankroll_context(result, selected_tier, bankroll_amount, bet_size)
    if ctx is None:
        return None

    if not ctx["has_bankroll_inputs"]:
        return (
            f"Bankroll Desk view: this is a {ctx['pressure']} spot, and the selected action is worth {ctx['best_ev']:+.3f} units per original bet. "
            f"{ctx['margin_text']}\n\n"
            "To make this analysis useful in real bankroll terms, enter the bankroll amount and current bet size. "
            "That unlocks guidance on how aggressive the stake is, what a reasonable unit size looks like, and how to stay solvent long enough for an edge to matter."
        )

    return (
        f"With a bankroll of ${ctx['bankroll_amount']:,.2f} and a live bet of ${ctx['bet_size']:,.2f}, you are risking "
        f"{ctx['bet_fraction'] * 100:.2f}% of bankroll on this hand, which is {ctx['stake_note']}. The selected action is {ctx['action']} at "
        f"{ctx['best_ev']:+.3f} units per original bet. {ctx['margin_text']} A disciplined unit for this bankroll is roughly "
        f"${ctx['conservative_unit']:,.2f} to ${ctx['assertive_unit']:,.2f} per hand, so this wager should be judged against that range.\n\n"
        f"{ctx['long_run_note']} To turn a profit in the long run, the player needs two things working together: a real edge in the decision "
        "and a bet size small enough that variance does not knock the bankroll out before the edge has time to compound. "
        "Bankroll discipline is what lets mathematical edge survive long enough to show up in actual results."
    )


def render_hero():
    st.markdown(
        """
        <section class="hero-shell">
            <div class="eyebrow">Table Sense, Not Just Table Math</div>
            <h1 class="hero-title">Blackjack AI Coach</h1>
            <p class="hero-copy">
                This interface teaches the hand, not just the move. Feed it the cards,
                choose a coaching style, and it will compare the expected value of each legal play,
                then explain the strategic idea you are meant to remember next time.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def load_gpt_model(checkpoint_path="out/best_model.pt", meta_path="out/meta.json"):
    """
    Load trained nanoGPT model and tokenizer metadata.
    Returns:
        model, stoi, itos, device
    or:
        None, None, None, None
    """
    ckpt_file = Path(checkpoint_path)
    meta_file = Path(meta_path)

    if not ckpt_file.exists() or not meta_file.exists():
        return None, None, None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    stoi = meta["stoi"]
    itos = {int(k): v for k, v in meta["itos"].items()}

    checkpoint = torch.load(ckpt_file, map_location=device)
    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, stoi, itos, device


def encode_text(text, stoi):
    return [stoi[c] for c in text]


def decode_tokens(tokens, itos):
    return "".join([itos[i] for i in tokens])


def generate_gpt_explanation(prompt, max_new_tokens=100, temperature=0.8, top_k=30):
    """
    Generate explanation text from the trained nanoGPT model if available.
    Falls back to None if model/tokenizer are unavailable or prompt has unseen chars.
    """
    model, stoi, itos, device = load_gpt_model()
    if model is None:
        return None

    try:
        x = torch.tensor([encode_text(prompt, stoi)], dtype=torch.long, device=device)
    except KeyError:
        return None

    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    return decode_tokens(y[0].tolist(), itos)


def extract_reason_only(generated_text):
    """
    Tries to shorten GPT output to the most useful explanation.
    """
    if generated_text is None:
        return None

    if "Reason:" in generated_text:
        return generated_text.split("Reason:", 1)[-1].strip()

    return generated_text.strip()


# --------------------------------------------------
# Hero
# --------------------------------------------------
render_hero()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Coach Settings")
st.sidebar.caption("Tune the rules and the explanation style.")
can_double = st.sidebar.checkbox("Double allowed", value=True)
can_split = st.sidebar.checkbox("Split allowed", value=True)
dealer_hits_soft_17 = st.sidebar.checkbox("Dealer hits soft 17", value=False)
explanation_mode = st.sidebar.radio(
    "Coach Mode",
    list(COACH_MODES.keys()),
    help="Choose the output style: classical coaching, EV reasoning, or bankroll-aware guidance.",
    key="coach_mode",
    on_change=sync_plan_from_coach_mode,
)
use_gpt = st.sidebar.checkbox("Use trained nanoGPT explanation if available", value=True)

st.sidebar.markdown("---")
st.sidebar.write("Recommended training checkpoint paths:")
st.sidebar.code("out/best_model.pt\nout/meta.json")


# --------------------------------------------------
# Intro panels
# --------------------------------------------------
intro_col1, intro_col2 = st.columns([1.2, 1])
with intro_col1:
    render_panel(
        "Coach Lens",
        "Expected value first",
        "The engine now compares legal plays by long-run units won or lost per original bet, then wraps the best financial choice in coaching language.",
        strong=True,
    )
with intro_col2:
    selected_plan = st.selectbox(
        "Membership Tier",
        list(PLAN_TIERS.keys()),
        help="Choose the plan level this version would sell to customers.",
        key="membership_tier",
        on_change=sync_coach_mode_from_plan,
    )
    selected_tier = PLAN_TIERS[selected_plan]
    render_tier_summary(selected_tier)


# --------------------------------------------------
# Input form
# --------------------------------------------------
left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
    st.markdown("## Deal The Hand")
    form_col1, form_col2, form_col3 = st.columns(3)

    with form_col1:
        player_card_1 = st.selectbox("Player Card 1", VALID_CARDS, index=7)
    with form_col2:
        player_card_2 = st.selectbox("Player Card 2", VALID_CARDS, index=7)
    with form_col3:
        dealer_card = st.selectbox("Dealer Upcard", VALID_CARDS, index=4)

    additional_cards_input = st.text_input(
        "Additional player cards",
        value="",
        help="Use this if the player already hit and has more than two cards. Example: 5,3",
        placeholder="Optional, comma-separated",
    )

    bankroll_amount = None
    bet_size = None
    if selected_tier["level"] == "elite":
        bankroll_col1, bankroll_col2 = st.columns(2)
        with bankroll_col1:
            bankroll_amount = st.number_input(
                "Bankroll ($)",
                min_value=1.0,
                value=1000.0,
                step=50.0,
                help="Total bankroll available for this session or bankroll plan.",
            )
        with bankroll_col2:
            bet_size = st.number_input(
                "Current Bet ($)",
                min_value=1.0,
                value=25.0,
                step=5.0,
                help="Current amount at risk on this hand before doubles or splits.",
            )

    analyze_button = st.button("Analyze Hand")

with right_col:
    render_panel(
        "What Makes It Feel Smart",
        "A coach should do more than name the move",
        "Strong blackjack teaching starts with the best long-run financial decision, then explains the pattern clearly enough to recognize it next time.",
    )
    st.markdown(
        """
        <div class="panel">
            <div class="section-kicker">Coach Modes</div>
            <div class="chips-row">
                <span class="chip">Table Coach: classical</span>
                <span class="chip">EV Edge: expected value</span>
                <span class="chip">Bankroll Desk: risk lens</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Main analysis
# --------------------------------------------------
if analyze_button:
    try:
        player_cards = [player_card_1, player_card_2]

        if additional_cards_input.strip():
            extras = [c.strip().upper() for c in additional_cards_input.split(",") if c.strip()]
            player_cards.extend(extras)

        result = recommend_action(
            player_cards=player_cards,
            dealer_card=dealer_card,
            can_double=can_double,
            can_split=can_split,
            dealer_hits_soft_17=dealer_hits_soft_17,
            deck_count=2,
        )

        action = result["recommended_action"]
        confidence = result["confidence"]
        best_ev = result.get("best_ev")
        ev_margin = result.get("ev_margin")
        explanation = result["explanation"]
        coach = result.get("coach", {})
        coach_text = selected_coach_text(coach, explanation_mode)
        if selected_tier["level"] == "free":
            coach_text = explanation
        elif explanation_mode == "Bankroll Desk":
            coach_text = bankroll_coach_call(result, selected_tier, bankroll_amount, bet_size)

        hand_type_parts = []
        if result["hand_info"]["is_soft"]:
            hand_type_parts.append("Soft")
        else:
            hand_type_parts.append("Hard")

        if result["hand_info"]["is_pair"]:
            hand_type_parts.append("Pair")

        if result["hand_info"]["is_blackjack"]:
            hand_type_parts.append("Blackjack")

        st.markdown("## Coach Recommendation")
        if selected_tier["level"] == "free":
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            metrics_col1.metric("Best Move", action)
            metrics_col2.metric("Model", "Classical")
            metrics_col3.metric("Player Total", result["hand_info"]["total"])
        else:
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            metrics_col1.metric("Best Move", action)
            metrics_col2.metric("Expected Value", f"{best_ev:+.3f}" if best_ev is not None else "N/A")
            metrics_col3.metric("EV Margin", f"{ev_margin:+.3f}" if ev_margin is not None else "N/A")
            metrics_col4.metric("Player Total", result["hand_info"]["total"])

        st.markdown(
            f"""
            <section class="coach-call">
                <div class="coach-call-label">Coach Call - {escape_text(selected_tier["model"])}</div>
                <p class="coach-call-text">{escape_text(coach_text or explanation)}</p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="chips-row" style="margin-bottom: 1rem;">
                <span class="chip">Player: {escape_text(normalize_for_display(result['state']['player_cards']))}</span>
                <span class="chip">Dealer: {escape_text(result['state']['dealer_card'])}</span>
                <span class="chip">Hand Type: {escape_text(' / '.join(hand_type_parts))}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        insight_col1, insight_col2, insight_col3 = st.columns(3)
        with insight_col1:
            if selected_tier["level"] == "free":
                render_insight_card("Why this is the move", explanation)
            else:
                render_insight_card("Why this is the move", coach.get("decision_summary", explanation))
        with insight_col2:
            if selected_tier["level"] == "free":
                render_insight_card("Pattern to remember", coach.get("teaching_tip", explanation))
            else:
                render_insight_card("Common mistake", coach.get("common_mistake", explanation))
        with insight_col3:
            if selected_tier["level"] == "elite":
                render_insight_card(
                    "Bankroll lens",
                    bankroll_lens_text(result, selected_tier, bankroll_amount, bet_size),
                )
            else:
                render_insight_card("Teaching tip", coach.get("teaching_tip", explanation))

        if selected_tier["level"] == "free":
            render_panel(
                "Classical View",
                "Readable table coaching",
                "This tier gives the move and the strategic pattern to remember. Upgrade tiers reveal the EV spread behind the recommendation.",
                strong=True,
            )
        else:
            render_panel(
                "Expected Value View",
                "Long-run financial logic",
                coach.get("math", explanation),
                strong=True,
            )

        if selected_tier["level"] == "elite":
            render_panel(
                "Bankroll Desk View",
                "Risk-aware decision framing",
                bankroll_guidance(result, selected_tier, bankroll_amount, bet_size),
                strong=True,
            )

        action_evs = result.get("action_evs", {})
        if action_evs and selected_tier["level"] in ["pro", "elite"]:
            ev_lines = "\n".join(
                f"{action_name}: {ev_value:+.3f} units"
                for action_name, ev_value in sorted(
                    action_evs.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
            with st.expander("Show EV comparison"):
                render_copy_block(ev_lines)
        elif action_evs:
            with st.expander("Upgrade preview"):
                render_copy_block("EV Edge unlocks action-by-action expected value comparisons.")

        gpt_prompt = format_for_gpt(result)

        if use_gpt:
            gpt_output = generate_gpt_explanation(
                prompt=gpt_prompt,
                max_new_tokens=120,
                temperature=0.8,
                top_k=30,
            )

            if gpt_output is None:
                render_panel(
                    "nanoGPT Coach Voice",
                    "Model output unavailable",
                    "No trained nanoGPT checkpoint was found, or the prompt contains characters outside the tokenizer. The EV-based coach explanation is still active.",
                )
            else:
                render_panel(
                    "nanoGPT Coach Voice",
                    "Additional natural-language flavor",
                    extract_reason_only(gpt_output),
                )

                with st.expander("Show full nanoGPT output"):
                    render_copy_block(gpt_output)

        if selected_tier["level"] in ["pro", "elite"]:
            with st.expander("Show EV engine prompt and raw output"):
                st.markdown("**Structured Prompt Sent to GPT**")
                render_copy_block(gpt_prompt)
                st.markdown("**Raw Engine Explanation**")
                st.write(explanation)
                st.markdown("**Raw Engine Output**")
                render_json_block(result)

    except Exception as e:
        st.error(f"Error: {e}")


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
footer_col1, footer_col2 = st.columns([1.15, 0.85], gap="large")

with footer_col1:
    render_panel(
        "How It Works",
        "Expected value first, coach framing on top",
        "The engine compares legal actions by long-run units won or lost per original bet. The UI then wraps the best EV choice in coaching language so the decision is financially grounded and easier to remember.",
    )

with footer_col2:
    with st.expander("Example test hands"):
        render_copy_block(
            "Player: 8,8 | Dealer: 6\n"
            "Player: 10,6 | Dealer: 7\n"
            "Player: A,7 | Dealer: 9\n"
            "Player: 5,5 | Dealer: 6"
        )
