from collections import Counter
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blackjack_engine import format_for_gpt, is_bust, recommend_action


RANKS: List[str] = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
NON_TEN_RANKS = {"A", "2", "3", "4", "5", "6", "7", "8", "9"}
TWO_DECK_LIMITS = {rank: 8 for rank in NON_TEN_RANKS}
TWO_DECK_LIMITS.update({"10": 8, "J": 8, "Q": 8, "K": 8})

MAX_PLAYER_CARDS = 4
OUTPUT_PATH = SCRIPT_DIR / "data" / "train.txt"
RANDOM_SEED = 1337
OPENING_HAND_SAMPLE_LIMIT = 60
LATER_HAND_SAMPLE_LIMIT = 16
LATER_HAND_ATTEMPTS_MULTIPLIER = 10
VARIANTS_PER_STATE = 3

ANCHOR_STATES: List[Tuple[Tuple[str, ...], str]] = [
    (("8", "8"), "6"),
    (("10", "6"), "7"),
    (("A", "7"), "9"),
    (("9", "9"), "7"),
    (("10", "2"), "4"),
    (("A", "6"), "3"),
    (("10", "10"), "10"),

    # Hard total anchors
    (("10", "5"), "10"),
    (("9", "7"), "10"),
    (("10", "3"), "6"),
    (("8", "4"), "6"),
    (("7", "5"), "2"),
    (("7", "3"), "9"),
    (("6", "5"), "10"),
    (("10", "4"), "6"),
    (("9", "5"), "6"),
    (("9", "4"), "2"),

    # Soft total anchors
    (("A", "5"), "4"),
    (("A", "8"), "6"),
    (("A", "2"), "5"),
    (("A", "3"), "6"),
    (("A", "4"), "4"),
    (("A", "7"), "2"),
    (("9", "2"), "5"),

    # Multi-card anchors
    (("5", "3", "3"), "6"),
    (("2", "2", "2", "9"), "7"),
    (("10", "2", "2"), "4"),
]

CURATED_PAIR_KNOWLEDGE = {
    (("2", "2"), "5"): {
        "Table Coach": "Split is the clean table play here. Against dealer 5, turning 2,2 into two fresh hands is stronger than keeping a weak 4 together.",
        "EV Edge": "Expected value view: Split is the better line against dealer 5 because a paired 2,2 has more value as two starting hands than as one weak hard 4.",
        "Bankroll Desk": "Split is the bankroll-aware play. This is the kind of dealer weakness where pressing the correct structure matters more than clinging to a low total.",
    },
    (("2", "2"), "3"): {
        "Table Coach": "Split here. Dealer 3 gives small pairs room to grow, and two new hands usually outperform a hard 4.",
        "EV Edge": "The EV model leans to Split against dealer 3 because keeping 2,2 together leaves too little total strength on the table.",
        "Bankroll Desk": "For bankroll play, Split is the right structure. The edge comes from creating two hands in a favorable dealer matchup instead of nursing a dead low total.",
    },
    (("3", "3"), "4"): {
        "Table Coach": "Split is the table play. Dealer 4 is weak enough that 3,3 works better as two hands than as one hard 6.",
        "EV Edge": "Expected value view: Split has the better long-run return here because dealer 4 lets two fresh hands capture more upside than standing on a weak 6.",
        "Bankroll Desk": "Split is the bankroll-aware choice. This is a measured spot to use the dealer's weakness instead of settling for a fragile hard 6.",
    },
    (("4", "4"), "5"): {
        "Table Coach": "Split is the clean play. Dealer 5 is weak, and two new hands usually give you more ways to profit than sitting on hard 8.",
        "EV Edge": "The EV edge favors Split versus dealer 5 because the upside of two playable hands beats the limited value of a stuck hard 8.",
        "Bankroll Desk": "Split is the bankroll-aware move. In a dealer-weak spot like this, the gain comes from structuring the hand for upside without overbetting elsewhere.",
    },
    (("5", "5"), "6"): {
        "Table Coach": "Do not split 5,5 here. Double is the stronger play because hard 10 is already a powerful total against dealer 6.",
        "EV Edge": "Expected value view: Double leads because hard 10 against dealer 6 is already a premium betting spot, and splitting would throw away that strength.",
        "Bankroll Desk": "Double is the bankroll-aware play. The edge comes from pressing a strong hard 10, not from breaking it into two weaker hands.",
    },
    (("6", "6"), "5"): {
        "Table Coach": "Split is the clean table answer. Dealer 5 is weak enough that 6,6 plays better as two hands than as one awkward 12.",
        "EV Edge": "The EV model prefers Split because hard 12 is clumsy, while dealer 5 gives two new hands room to outperform it.",
        "Bankroll Desk": "Split is the bankroll-aware route. This is a dealer-weak setup where creating two hands is worth more than defending a shaky 12.",
    },
    (("7", "7"), "6"): {
        "Table Coach": "Split here. Dealer 6 is weak, so 7,7 gains more value as two hands than as a hard 14.",
        "EV Edge": "Expected value view: Split outruns the alternatives because a hard 14 is too stiff, while dealer 6 gives two fresh hands a real edge.",
        "Bankroll Desk": "Split is the bankroll-aware play. The profit comes from using the dealer's weakness to turn a stiff total into two workable starts.",
    },
    (("8", "8"), "10"): {
        "Table Coach": "Split anyway. Hard 16 against a dealer 10 is so poor that breaking up 8,8 is still the better path.",
        "EV Edge": "The EV model still prefers Split because even though dealer 10 is strong, keeping hard 16 together is one of the worst long-run holdings.",
        "Bankroll Desk": "Split is still the bankroll-aware choice. This is damage control more than aggression, but the split structure loses less often over time than standing pat on 16.",
    },
    (("9", "9"), "10"): {
        "Table Coach": "Stand here. Against dealer 10, 18 is already strong enough that splitting 9,9 usually gives away value.",
        "EV Edge": "Expected value view: Stand leads because made 18 holds more long-run value than turning the hand into two uncertain starts against dealer 10.",
        "Bankroll Desk": "Stand is the bankroll-aware play. When you already have 18 against a strong dealer card, preserving the made hand usually beats chasing extra variance.",
    },
    (("A", "A"), "6"): {
        "Table Coach": "Split aces here. Two fresh ace-starting hands are much stronger than trying to play A,A as a soft 12.",
        "EV Edge": "The EV edge strongly favors Split because A,A has far more long-run value as two new hands than as one soft 12 against dealer 6.",
        "Bankroll Desk": "Split is the bankroll-aware answer. This is one of the clearest spots to create two high-upside hands instead of trapping value in a weak total.",
    },
}

RULE_VARIATION_STATES: List[Dict[str, object]] = [
    {
        "player_cards": ("5", "5"),
        "dealer_card": "6",
        "can_double": False,
        "can_split": True,
        "dealer_hits_soft_17": False,
        "deck_count": 2,
    },
    {
        "player_cards": ("8", "8"),
        "dealer_card": "6",
        "can_double": True,
        "can_split": False,
        "dealer_hits_soft_17": False,
        "deck_count": 2,
    },
    {
        "player_cards": ("A", "6"),
        "dealer_card": "2",
        "can_double": True,
        "can_split": True,
        "dealer_hits_soft_17": True,
        "deck_count": 2,
    },
    {
        "player_cards": ("A", "7"),
        "dealer_card": "2",
        "can_double": False,
        "can_split": True,
        "dealer_hits_soft_17": False,
        "deck_count": 2,
    },
    {
        "player_cards": ("9", "9"),
        "dealer_card": "7",
        "can_double": True,
        "can_split": True,
        "dealer_hits_soft_17": True,
        "deck_count": 2,
    },
    {
        "player_cards": ("10", "6"),
        "dealer_card": "7",
        "can_double": False,
        "can_split": False,
        "dealer_hits_soft_17": False,
        "deck_count": 2,
    },
]


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
    counts = Counter(cards)
    return all(counts[rank] <= TWO_DECK_LIMITS[rank] for rank in counts)


def canonical_hand(cards: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(cards, key=lambda rank: (RANKS.index(rank), rank)))


def valid_state(player_cards: Sequence[str], dealer_card: str) -> bool:
    all_cards = list(player_cards) + [dealer_card]
    return cards_within_two_decks(all_cards)


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
        return random.choice([variant for variant in variants if variant])

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


def serialize_state(
    player_cards: Sequence[str],
    dealer_card: str,
    *,
    can_double: bool = True,
    can_split: bool = True,
    dealer_hits_soft_17: bool = False,
    deck_count: int = 2,
) -> List[str]:
    result = recommend_action(
        player_cards=list(player_cards),
        dealer_card=dealer_card,
        can_double=can_double,
        can_split=can_split,
        dealer_hits_soft_17=dealer_hits_soft_17,
        deck_count=deck_count,
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


def random_hand(card_count: int) -> Tuple[str, ...]:
    return canonical_hand(random.choices(RANKS, k=card_count))


def opening_states() -> List[Tuple[Tuple[str, ...], str]]:
    seen = set()
    attempts = 0
    max_attempts = OPENING_HAND_SAMPLE_LIMIT * 6

    while len(seen) < OPENING_HAND_SAMPLE_LIMIT and attempts < max_attempts:
        attempts += 1
        hand = random_hand(2)
        dealer_card = random.choice(RANKS)
        if not valid_state(hand, dealer_card):
            continue
        if len(set(hand)) == 1:
            continue
        seen.add((hand, dealer_card))

    print(f"Collected {len(seen)} sampled opening-hand states after {attempts} attempts.")
    return sorted(seen)


def later_hit_states() -> List[Tuple[Tuple[str, ...], str]]:
    sampled_states: List[Tuple[Tuple[str, ...], str]] = []

    for card_count in range(3, MAX_PLAYER_CARDS + 1):
        seen = set()
        attempts = 0
        max_attempts = LATER_HAND_SAMPLE_LIMIT * LATER_HAND_ATTEMPTS_MULTIPLIER

        while len(seen) < LATER_HAND_SAMPLE_LIMIT and attempts < max_attempts:
            attempts += 1
            hand = random_hand(card_count)
            dealer_card = random.choice(RANKS)

            if not valid_state(hand, dealer_card):
                continue

            if is_bust(list(hand)):
                continue

            seen.add((hand, dealer_card))

        sampled_states.extend(sorted(seen))
        print(
            f"Collected {len(seen)} sampled later-hand states "
            f"for {card_count}-card player hands after {attempts} attempts."
        )

    return sampled_states


def build_dataset_lines() -> List[str]:
    serialized_lines = set()

    print("Building anchor training states...")
    for player_cards, dealer_card in ANCHOR_STATES:
        serialized_lines.update(serialize_state(player_cards, dealer_card))

    print("Building curated pair knowledge...")
    for (player_cards, dealer_card), tier_map in CURATED_PAIR_KNOWLEDGE.items():
        for tier_name, text in tier_map.items():
            tier_meta = TIER_PROMPTS[tier_name]
            serialized_lines.add(
                f"Player: {','.join(player_cards)} | Dealer: {dealer_card} | "
                f"Tier: {tier_name} | Voice: {tier_meta['mode']} | "
                f"{tier_meta['response_label']}: {text}"
            )

    print("Building rule-variation training states...")
    for state in RULE_VARIATION_STATES:
        serialized_lines.update(
            serialize_state(
                state["player_cards"],
                state["dealer_card"],
                can_double=state["can_double"],
                can_split=state["can_split"],
                dealer_hits_soft_17=state["dealer_hits_soft_17"],
                deck_count=state["deck_count"],
            )
        )

    print("Building opening-hand states...")
    for player_cards, dealer_card in opening_states():
        serialized_lines.update(serialize_state(player_cards, dealer_card))

    print("Building sampled later-hit states...")
    for player_cards, dealer_card in later_hit_states():
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
