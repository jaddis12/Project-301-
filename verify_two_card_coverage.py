from itertools import combinations_with_replacement
from pathlib import Path
import json
import sys
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blackjack_engine import format_for_gpt, recommend_action


RANKS: List[str] = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
NON_TEN_RANKS = {"A", "2", "3", "4", "5", "6", "7", "8", "9"}
TWO_DECK_LIMITS = {rank: 8 for rank in NON_TEN_RANKS}
TWO_DECK_LIMITS.update({"10": 8, "J": 8, "Q": 8, "K": 8})

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


def canonical_hand(cards: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(sorted(cards, key=lambda rank: (RANKS.index(rank), rank)))


def cards_within_two_decks(cards: Tuple[str, ...]) -> bool:
    counts: Dict[str, int] = {}
    for card in cards:
        counts[card] = counts.get(card, 0) + 1
    return all(counts[rank] <= TWO_DECK_LIMITS[rank] for rank in counts)


def load_lines() -> List[str]:
    train_path = SCRIPT_DIR / "data" / "train.txt"
    return train_path.read_text(encoding="utf-8").splitlines()


def load_stoi() -> Dict[str, int]:
    meta_path = SCRIPT_DIR / "out" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta["stoi"]


def build_exact_map(lines: List[str]) -> Dict[Tuple[str, str, str, str], str]:
    exact_map = {}
    for line in lines:
        tier_part = line.split(" | Tier: ", 1)[1]
        tier_name, rest = tier_part.split(" | Voice: ", 1)
        voice, rest = rest.split(" | ", 1)
        response_label, response_text = rest.split(": ", 1)
        prefix = line.split(f" | Tier: {tier_name} | Voice: {voice} | {response_label}:", 1)[0]
        exact_map[(prefix, tier_name, voice, response_label)] = response_text
    return exact_map


def readable(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 40:
        return False
    words = stripped.replace("|", " ").split()
    return len(words) >= 8


def all_two_card_states() -> List[Tuple[Tuple[str, ...], str]]:
    states = []
    for hand in combinations_with_replacement(RANKS, 2):
        canonical = canonical_hand(hand)
        for dealer_card in RANKS:
            if not cards_within_two_decks(canonical + (dealer_card,)):
                continue
            states.append((canonical, dealer_card))
    return states


def main() -> None:
    lines = load_lines()
    stoi = load_stoi()
    exact_map = build_exact_map(lines)

    parse_failures = 0
    encode_failures = 0
    unreadable = []
    missing = []

    for line in lines:
        try:
            _ = line.split(" | Tier: ", 1)[1]
        except Exception:
            parse_failures += 1
            continue
        try:
            for ch in line:
                _ = stoi[ch]
        except Exception:
            encode_failures += 1

    states = all_two_card_states()
    for player_cards, dealer_card in states:
        result = recommend_action(
            player_cards=list(player_cards),
            dealer_card=dealer_card,
            can_double=DEFAULT_RULES["can_double"],
            can_split=DEFAULT_RULES["can_split"],
            dealer_hits_soft_17=DEFAULT_RULES["dealer_hits_soft_17"],
            deck_count=DEFAULT_RULES["deck_count"],
        )
        base_prompt = format_for_gpt(result)

        for tier_name, tier_meta in TIER_PROMPTS.items():
            key = (base_prompt, tier_name, tier_meta["mode"], tier_meta["response_label"])
            response = exact_map.get(key)
            if response is None:
                missing.append((player_cards, dealer_card, tier_name))
                continue
            if not readable(response):
                unreadable.append((player_cards, dealer_card, tier_name, response))

    total_expected = len(states) * len(TIER_PROMPTS)
    found = total_expected - len(missing)

    print(f"TOTAL_DATASET_LINES: {len(lines)}")
    print(f"PARSE_FAILURES: {parse_failures}")
    print(f"ENCODE_FAILURES: {encode_failures}")
    print(f"TWO_CARD_STATES: {len(states)}")
    print(f"EXPECTED_TWO_CARD_TIER_RESPONSES: {total_expected}")
    print(f"FOUND_TWO_CARD_TIER_RESPONSES: {found}")
    print(f"MISSING_TWO_CARD_TIER_RESPONSES: {len(missing)}")
    print(f"UNREADABLE_TWO_CARD_TIER_RESPONSES: {len(unreadable)}")

    if missing:
        print("MISSING_SAMPLES:")
        for sample in missing[:10]:
            print(sample)

    if unreadable:
        print("UNREADABLE_SAMPLES:")
        for sample in unreadable[:10]:
            print(sample[:3], sample[3])


if __name__ == "__main__":
    main()
