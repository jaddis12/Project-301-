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
        color: rgba(255, 248, 239, 0.88);
        font-size: 1.04rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    .hero-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
    }

    .hero-pill {
        border: 1px solid rgba(255, 255, 255, 0.18);
        background: rgba(255, 255, 255, 0.10);
        border-radius: 999px;
        padding: 0.45rem 0.9rem;
        font-size: 0.88rem;
        color: #fff8ef;
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

    div[data-testid="metric-container"] {
        background: rgba(255, 252, 247, 0.86);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1rem 1rem 0.9rem 1rem;
        box-shadow: 0 10px 30px rgba(28, 45, 34, 0.06);
    }

    div[data-testid="metric-container"] label {
        color: var(--muted) !important;
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


def normalize_for_display(cards):
    return ",".join(cards)


def escape_text(text):
    return html.escape(str(text))


def selected_coach_text(coach_payload, explanation_mode):
    mode_to_key = {
        "Quick": "quick",
        "Beginner": "beginner",
        "Math": "math",
    }
    return coach_payload.get(mode_to_key[explanation_mode], coach_payload.get("beginner", ""))


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


def render_hero():
    st.markdown(
        """
        <section class="hero-shell">
            <div class="eyebrow">Table Sense, Not Just Table Math</div>
            <h1 class="hero-title">Blackjack AI Coach</h1>
            <p class="hero-copy">
                This interface should teach the hand, not just name the move. Feed it the cards,
                choose a coaching style, and it will explain the decision, the trap people fall into,
                and the strategic idea you are meant to remember next time.
            </p>
            <div class="hero-strip">
                <span class="hero-pill">Beginner mode for plain English</span>
                <span class="hero-pill">Quick mode for fast table decisions</span>
                <span class="hero-pill">Math mode for strategy reasoning</span>
            </div>
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
    ["Beginner", "Quick", "Math"],
    index=0,
    help="Choose how the coach explains the decision.",
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
        "Designed to explain the hand",
        "The primary output is the coaching language around the move: why it is right, what people misread, and what pattern to learn.",
        strong=True,
    )
with intro_col2:
    render_panel(
        "Mode Shift",
        f"Current explanation style: {explanation_mode}",
        "Switch modes in the sidebar to pivot between plain-English teaching, one-line speed, and math-oriented reasoning.",
    )


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

    analyze_button = st.button("Analyze Hand")

with right_col:
    render_panel(
        "What Makes It Feel Smart",
        "A coach should do more than name the move",
        "Strong blackjack teaching explains the decision, anticipates confusion, and reinforces the strategic pattern behind the spot.",
    )
    st.markdown(
        """
        <div class="panel">
            <div class="section-kicker">Coach Modes</div>
            <div class="chips-row">
                <span class="chip">Beginner: plain English</span>
                <span class="chip">Quick: one sentence</span>
                <span class="chip">Math: long-run logic</span>
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
        explanation = result["explanation"]
        coach = result.get("coach", {})
        coach_text = selected_coach_text(coach, explanation_mode)

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
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        metrics_col1.metric("Best Move", action)
        metrics_col2.metric("Coach Confidence", f"{confidence:.2f}")
        metrics_col3.metric("Player Total", result["hand_info"]["total"])

        st.markdown(
            f"""
            <section class="coach-call">
                <div class="coach-call-label">Coach Call · {escape_text(explanation_mode)} Mode</div>
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
            render_insight_card("Why this is the move", coach.get("decision_summary", explanation))
        with insight_col2:
            render_insight_card("Common mistake", coach.get("common_mistake", explanation))
        with insight_col3:
            render_insight_card("Teaching tip", coach.get("teaching_tip", explanation))

        render_panel(
            "Math View",
            "Long-run strategy logic",
            coach.get("math", explanation),
            strong=True,
        )

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
                    "No trained nanoGPT checkpoint was found, or the prompt contains characters outside the tokenizer. The rules-based coach explanation is still active.",
                )
            else:
                render_panel(
                    "nanoGPT Coach Voice",
                    "Additional natural-language flavor",
                    extract_reason_only(gpt_output),
                )

                with st.expander("Show full nanoGPT output"):
                    st.code(gpt_output)

        with st.expander("Show rules engine prompt and raw output"):
            st.markdown("**Structured Prompt Sent to GPT**")
            st.code(gpt_prompt)
            st.markdown("**Raw Engine Explanation**")
            st.write(explanation)
            st.markdown("**Raw Engine Output**")
            st.json(result)

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
        "Rules engine first, coach framing on top",
        "The engine still chooses the action. The UI now wraps that answer in coaching language so the user gets decision support, confusion handling, and strategy teaching in one place.",
    )

with footer_col2:
    with st.expander("Example test hands"):
        st.code(
            "Player: 8,8 | Dealer: 6\n"
            "Player: 10,6 | Dealer: 7\n"
            "Player: A,7 | Dealer: 9\n"
            "Player: 5,5 | Dealer: 6"
        )

    with st.expander("Project file checklist"):
        st.code(
            "model.py\n"
            "train.py\n"
            "sample.py\n"
            "blackjack_engine.py\n"
            "app.py\n"
            "data/train.txt\n"
            "out/best_model.pt\n"
            "out/meta.json"
        )
