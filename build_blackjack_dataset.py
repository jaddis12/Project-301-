from itertools import combinations_with_replacement
from pathlib import Path
import random
import sys
from typing import Dict, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blackjack_engine import format_for_gpt, recommend_action


RANKS: List[str] = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
NON_TEN_RANKS = {"A", "2", "3", "4", "5", "6", "7", "8", "9"}
TWO_DECK_LIMITS = {rank: 8 for rank in NON_TEN_RANKS}
TWO_DECK_LIMITS.update({"10": 8, "J": 8, "Q": 8, "K": 8})

OUTPUT_PATH = SCRIPT_DIR / "data" / "train.txt"
RANDOM_SEED = 1337
VARIANTS_PER_STATE = 3
DEFAULT_RULES = {
    "can_double": True,
    "can_split": True,
    "dealer_hits_soft_17": False,
    "deck_count": 2,
}

TIER_PROMPTS = {
    "Table Coach": {
        "mode": "classical",
        "response_label": "Coach Call",
    },
    "EV Edge": {
        "mode": "expected_value",
        "response_label": "Coach Call",
    },
    "Bankroll Desk": {
        "mode": "bankroll",
        "response_label": "Bankroll Lens",
    },
}


def format_best_ev(best_ev) -> str:
    if best_ev is None:
        return "N/A"
    return f"{best_ev:+.3f}"


def format_margin(ev_margin) -> str:
    if ev_margin is None:
        return "N/A"
    return f"{ev_margin:+.3f}"


def cards_within_two_decks(cards: Sequence[str]) -> bool:
    counts: Dict[str, int] = {}
    for card in cards:
        counts[card] = counts.get(card, 0) + 1
    return all(counts[rank] <= TWO_DECK_LIMITS[rank] for rank in counts)


def canonical_hand(cards: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted(cards, key=lambda rank: (RANKS.index(rank), rank)))


def valid_two_card_state(player_cards: Sequence[str], dealer_card: str) -> bool:
    return cards_within_two_decks(list(player_cards) + [dealer_card])


def tiered_response(result: dict, tier_name: str) -> str:
    coach = result.get("coach", {})
    action = result["recommended_action"]
    best_ev = result.get("best_ev")
    ev_margin = result.get("ev_margin")
    explanation = result["explanation"]
    math_reason = coach.get("math") or coach.get("decision_summary") or explanation
    decision_summary = coach.get("decision_summary") or explanation
    teaching_tip = coach.get("teaching_tip") or explanation
    common_mistake = coach.get("common_mistake") or explanation

    if tier_name == "Table Coach":
        variants = [
            coach.get("beginner") or explanation,
            f"{action}, because {decision_summary.lower()}",
            f"{action} is the clean table play here. {teaching_tip}",
            f"Stick with {action}. {common_mistake} {teaching_tip}",
        ]
        options = [variant for variant in variants if variant]
        return random.choice(options)

    if tier_name == "EV Edge":
        if best_ev is None:
            return math_reason
        variants = [
            (
                f"{math_reason} The best play is worth {format_best_ev(best_ev)} units per original bet. "
                f"The edge over the next-best option is {format_margin(ev_margin)} units."
            ),
            (
                f"Expected value view: {action} leads the board at {format_best_ev(best_ev)} units. "
                f"The separation from the runner-up is {format_margin(ev_margin)} units, so the math is fairly clear."
            ),
            (
                f"The EV model prefers {action}. It grades this choice at {format_best_ev(best_ev)} units per original bet, "
                f"with a margin of {format_margin(ev_margin)} over the next-best line."
            ),
            (
                f"{action} is the financially strongest play. {math_reason} "
                f"On the model, that comes out to {format_best_ev(best_ev)} units with a {format_margin(ev_margin)} unit cushion."
            ),
        ]
        return random.choice(variants)

    pressure = "thin" if best_ev is not None and best_ev <= 0 else "positive"
    if best_ev is None:
        variants = [
            (
                f"{action} is the bankroll-aware play. This is a {pressure} spot, so the job is to stay disciplined "
                "and avoid letting variance force bad betting decisions."
            ),
            (
                f"Bankroll lens: {action} keeps the hand aligned with long-run discipline. "
                "Without a strong edge, stake control matters just as much as the move itself."
            ),
        ]
        return random.choice(variants)

    variants = [
        (
            f"{action} is the bankroll-aware play. This is a {pressure} edge at {format_best_ev(best_ev)} units per original bet, "
            "so profit comes from pairing the right move with disciplined bet sizing and enough bankroll to survive variance."
        ),
        (
            f"Bankroll lens: {action} is the correct line, but the money is made only if the stake stays controlled. "
            f"The hand is worth {format_best_ev(best_ev)} units, which means sizing discipline is part of the edge."
        ),
        (
            f"{action} protects the bankroll better than the alternatives. At {format_best_ev(best_ev)} units of expectation, "
            "this is the kind of edge that compounds only when the player avoids oversized bets."
        ),
        (
            f"For bankroll play, {action} is the right decision. The model values the hand at {format_best_ev(best_ev)} units per original bet, "
            "so long-run profit depends on surviving variance long enough for that edge to repeat."
        ),
    ]
    return random.choice(variants)


def serialize_state(player_cards: Sequence[str], dealer_card: str) -> List[str]:
    result = recommend_action(
        player_cards=list(player_cards),
        dealer_card=dealer_card,
        can_double=DEFAULT_RULES["can_double"],
        can_split=DEFAULT_RULES["can_split"],
        dealer_hits_soft_17=DEFAULT_RULES["dealer_hits_soft_17"],
        deck_count=DEFAULT_RULES["deck_count"],
    )
    base_prompt = format_for_gpt(result)
    lines = []
    for _ in range(VARIANTS_PER_STATE):
        for tier_name, tier_meta in TIER_PROMPTS.items():
            lines.append(
                f"{base_prompt} | Tier: {tier_name} | Voice: {tier_meta['mode']} | "
                f"{tier_meta['response_label']}: {tiered_response(result, tier_name)}"
            )
    return lines


def all_two_card_states() -> List[Tuple[Tuple[str, ...], str]]:
    states: List[Tuple[Tuple[str, ...], str]] = []
    for hand in combinations_with_replacement(RANKS, 2):
        canonical = canonical_hand(hand)
        for dealer_card in RANKS:
            if not valid_two_card_state(canonical, dealer_card):
                continue
            states.append((canonical, dealer_card))
    print(f"Prepared {len(states)} exhaustive two-card states.")
    return states


def build_dataset_lines() -> List[str]:
    serialized_lines = set()
    print("Building exhaustive two-card training states...")
    for player_cards, dealer_card in all_two_card_states():
        serialized_lines.update(serialize_state(player_cards, dealer_card))
    print(f"Prepared {len(serialized_lines)} unique serialized examples.")
    return sorted(serialized_lines)


def main() -> None:
    random.seed(RANDOM_SEED)
    lines = build_dataset_lines()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} training examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
