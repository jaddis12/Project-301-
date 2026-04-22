# blackjack_engine.py
# Basic two-deck blackjack decision engine for an MVP.
#
# What this file does:
# - evaluates blackjack hands
# - detects soft totals and pairs
# - recommends an action using standard two-deck basic-strategy-style rules
# - returns a structured explanation payload for your GPT layer
#
# Supported actions:
# - "Hit"
# - "Stand"
# - "Double"
# - "Split"
#
# Notes:
# - This is a rules-based MVP engine, not a full casino EV simulator.
# - It assumes dealer stands on soft 17 by default unless changed in rules.
# - Surrender and insurance are omitted for simplicity.
# - After splitting aces / resplitting rules are not fully modeled here.
#
# Example:
#   state = GameState(
#       player_cards=["8", "8"],
#       dealer_card="6",
#       can_double=True,
#       can_split=True,
#   )
#   result = best_action(state)
#   print(result["recommended_action"])
#   print(result["explanation"])

from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import List, Dict, Any, Tuple


# --------------------------------------------------
# Card utilities
# --------------------------------------------------
RANK_VALUES = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 10,
    "Q": 10,
    "K": 10,
    "A": 11,
}

RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")


def normalize_card(card: str) -> str:
    """Normalize card input into supported ranks."""
    card = str(card).strip().upper()
    if card in {"T"}:
        return "10"
    if card in {"J", "Q", "K", "A", "2", "3", "4", "5", "6", "7", "8", "9", "10"}:
        return card
    raise ValueError(f"Unsupported card rank: {card}")


def card_value(card: str) -> int:
    """Return blackjack value for a single card rank."""
    return RANK_VALUES[normalize_card(card)]


def dealer_upcard_value(card: str) -> int:
    """Dealer upcard numeric value used by strategy tables."""
    return card_value(card)


# --------------------------------------------------
# Data model
# --------------------------------------------------
@dataclass
class GameState:
    player_cards: List[str]
    dealer_card: str
    can_double: bool = True
    can_split: bool = True
    dealer_hits_soft_17: bool = False
    deck_count: int = 2

    def normalized(self) -> "GameState":
        return GameState(
            player_cards=[normalize_card(c) for c in self.player_cards],
            dealer_card=normalize_card(self.dealer_card),
            can_double=self.can_double,
            can_split=self.can_split,
            dealer_hits_soft_17=self.dealer_hits_soft_17,
            deck_count=self.deck_count,
        )


# --------------------------------------------------
# Hand evaluation
# --------------------------------------------------
def hand_value(cards: List[str]) -> int:
    """
    Returns the best blackjack hand total <= 21 if possible.
    Handles aces as 11 or 1.
    """
    cards = [normalize_card(c) for c in cards]
    total = sum(card_value(c) for c in cards)
    ace_count = sum(1 for c in cards if c == "A")

    while total > 21 and ace_count > 0:
        total -= 10
        ace_count -= 1

    return total


def is_soft_hand(cards: List[str]) -> bool:
    """
    True if the hand contains an ace counted as 11 in the final best total.
    """
    cards = [normalize_card(c) for c in cards]
    total = sum(card_value(c) for c in cards)
    ace_count = sum(1 for c in cards if c == "A")

    while total > 21 and ace_count > 0:
        total -= 10
        ace_count -= 1

       # Any ace not converted from 11 down to 1 means the hand remains soft.
    return ace_count > 0 and total <= 21


def soft_total(cards: List[str]) -> int:
    """
    Returns the 'soft total' label for a soft hand.
    Example: A,6 -> 17
    """
    return hand_value(cards)


def is_pair(cards: List[str]) -> bool:
    """True if exactly two cards of equal split value are present."""
    if len(cards) != 2:
        return False
    c1 = normalize_card(cards[0])
    c2 = normalize_card(cards[1])

    # Treat all 10-value cards as equivalent for splitting logic only if same value.
    # Standard splitting usually requires actual same rank in real casinos,
    # but for an MVP we use equal blackjack value.
    return card_value(c1) == card_value(c2)


def is_blackjack(cards: List[str]) -> bool:
    """Natural blackjack: exactly two cards totaling 21."""
    return len(cards) == 2 and hand_value(cards) == 21


def is_bust(cards: List[str]) -> bool:
    return hand_value(cards) > 21


# --------------------------------------------------
# Strategy tables
# --------------------------------------------------
def pair_action(pair_rank: str, dealer: int, can_double: bool = True) -> str:
    """
    Two-deck-inspired pair strategy for MVP.
    Assumes split is available.
    """
    pair_rank = normalize_card(pair_rank)

    # Convert all 10-value ranks into "10"
    if card_value(pair_rank) == 10 and pair_rank != "A":
        pair_rank = "10"

    if pair_rank == "A":
        return "Split"
    if pair_rank == "10":
        return "Stand"
    if pair_rank == "9":
        if dealer in [2, 3, 4, 5, 6, 8, 9]:
            return "Split"
        return "Stand"
    if pair_rank == "8":
        return "Split"
    if pair_rank == "7":
        if dealer in [2, 3, 4, 5, 6, 7]:
            return "Split"
        return "Hit"
    if pair_rank == "6":
        if dealer in [2, 3, 4, 5, 6]:
            return "Split"
        return "Hit"
    if pair_rank == "5":
        return hard_total_action(10, dealer, can_double=can_double)
    if pair_rank == "4":
        if dealer in [5, 6]:
            return "Split"
        return "Hit"
    if pair_rank in ["3", "2"]:
        if dealer in [2, 3, 4, 5, 6, 7]:
            return "Split"
        return "Hit"

    return "Hit"


def soft_hand_action(total: int, dealer: int, can_double: bool = True) -> str:
    """
    Soft total basic strategy for two-card soft hands.
    total should be 13..20 typically (A,2 through A,9).
    """
    if total in [20, 19]:
        # Some charts double soft 19 vs 6 in some games; keep MVP simple.
        return "Stand"

    if total == 18:
        if dealer in [3, 4, 5, 6] and can_double:
            return "Double"
        if dealer in [2, 7, 8]:
            return "Stand"
        return "Hit"

    if total == 17:
        if dealer in [3, 4, 5, 6] and can_double:
            return "Double"
        return "Hit"

    if total in [15, 16]:
        if dealer in [4, 5, 6] and can_double:
            return "Double"
        return "Hit"

    if total in [13, 14]:
        if dealer in [5, 6] and can_double:
            return "Double"
        return "Hit"

    return "Hit"


def hard_total_action(total: int, dealer: int, can_double: bool = True) -> str:
    """
    Hard total basic strategy.
    """
    if total >= 17:
        return "Stand"

    if total in [13, 14, 15, 16]:
        if dealer in [2, 3, 4, 5, 6]:
            return "Stand"
        return "Hit"

    if total == 12:
        if dealer in [4, 5, 6]:
            return "Stand"
        return "Hit"

    if total == 11:
        return "Double" if can_double else "Hit"

    if total == 10:
        if dealer in [2, 3, 4, 5, 6, 7, 8, 9]:
            return "Double" if can_double else "Hit"
        return "Hit"

    if total == 9:
        if dealer in [3, 4, 5, 6]:
            return "Double" if can_double else "Hit"
        return "Hit"

    return "Hit"


# --------------------------------------------------
# Expected-value model
# --------------------------------------------------
def fresh_deck_counts(deck_count: int = 2) -> Tuple[int, ...]:
    """
    Return a finite-deck rank-count tuple.

    The model tracks rank identity so pair/split inputs remain clear, while
    10/J/Q/K all evaluate to value 10.
    """
    deck_count = max(1, int(deck_count))
    return tuple(4 * deck_count for _ in RANKS)


def remove_card_from_counts(deck_counts: Tuple[int, ...], card: str) -> Tuple[int, ...]:
    card = normalize_card(card)
    index = RANKS.index(card)
    counts = list(deck_counts)
    if counts[index] <= 0:
        raise ValueError(f"Too many visible {card} cards for this deck count.")
    counts[index] -= 1
    return tuple(counts)


def remove_visible_cards(deck_counts: Tuple[int, ...], cards: List[str]) -> Tuple[int, ...]:
    for card in cards:
        deck_counts = remove_card_from_counts(deck_counts, card)
    return deck_counts


def draw_probabilities(deck_counts: Tuple[int, ...]):
    total_cards = sum(deck_counts)
    if total_cards <= 0:
        return

    for index, count in enumerate(deck_counts):
        if count <= 0:
            continue

        next_counts = list(deck_counts)
        next_counts[index] -= 1
        yield RANKS[index], count / total_cards, tuple(next_counts)


def total_and_soft_from_cards(cards: Tuple[str, ...]) -> Tuple[int, bool]:
    total = sum(card_value(card) for card in cards)
    soft_aces = sum(1 for card in cards if normalize_card(card) == "A")

    while total > 21 and soft_aces > 0:
        total -= 10
        soft_aces -= 1

    return total, soft_aces > 0 and total <= 21


def total_and_soft_from_state(total: int, soft_aces: int) -> Tuple[int, int]:
    while total > 21 and soft_aces > 0:
        total -= 10
        soft_aces -= 1
    return total, soft_aces


def add_card_to_total(total: int, soft_aces: int, card: str) -> Tuple[int, int]:
    card = normalize_card(card)
    total += card_value(card)
    if card == "A":
        soft_aces += 1
    return total_and_soft_from_state(total, soft_aces)


def should_dealer_hit(total: int, soft_aces: int, dealer_hits_soft_17: bool) -> bool:
    if total < 17:
        return True
    if total == 17 and soft_aces > 0 and dealer_hits_soft_17:
        return True
    return False


@lru_cache(maxsize=None)
def dealer_finish_distribution(
    total: int,
    soft_aces: int,
    deck_counts: Tuple[int, ...],
    dealer_hits_soft_17: bool,
) -> Tuple[Tuple[str, float], ...]:
    """
    Dealer final distribution from a current dealer total.
    Keys are "bust" or string totals like "17".
    """
    total, soft_aces = total_and_soft_from_state(total, soft_aces)

    if total > 21:
        return (("bust", 1.0),)

    if not should_dealer_hit(total, soft_aces, dealer_hits_soft_17):
        return ((str(total), 1.0),)

    outcomes: Dict[str, float] = {}
    for card, probability, next_counts in draw_probabilities(deck_counts):
        next_total, next_soft_aces = add_card_to_total(total, soft_aces, card)
        for outcome, outcome_probability in dealer_finish_distribution(
            next_total,
            next_soft_aces,
            next_counts,
            dealer_hits_soft_17,
        ):
            outcomes[outcome] = outcomes.get(outcome, 0.0) + probability * outcome_probability

    return tuple(sorted(outcomes.items()))


@lru_cache(maxsize=None)
def dealer_distribution_from_upcard(
    dealer_card: str,
    deck_counts: Tuple[int, ...],
    dealer_hits_soft_17: bool,
) -> Tuple[Tuple[str, float], ...]:
    """
    Dealer final distribution after drawing the hidden card and completing play.
    """
    dealer_card = normalize_card(dealer_card)
    base_total = card_value(dealer_card)
    base_soft_aces = 1 if dealer_card == "A" else 0
    outcomes: Dict[str, float] = {}

    for hole_card, probability, next_counts in draw_probabilities(deck_counts):
        total, soft_aces = add_card_to_total(base_total, base_soft_aces, hole_card)
        for outcome, outcome_probability in dealer_finish_distribution(
            total,
            soft_aces,
            next_counts,
            dealer_hits_soft_17,
        ):
            outcomes[outcome] = outcomes.get(outcome, 0.0) + probability * outcome_probability

    return tuple(sorted(outcomes.items()))


def compare_player_to_dealer(player_total: int, dealer_outcome: str) -> float:
    if player_total > 21:
        return -1.0
    if dealer_outcome == "bust":
        return 1.0

    dealer_total = int(dealer_outcome)
    if player_total > dealer_total:
        return 1.0
    if player_total < dealer_total:
        return -1.0
    return 0.0


@lru_cache(maxsize=None)
def stand_ev(
    cards: Tuple[str, ...],
    dealer_card: str,
    deck_counts: Tuple[int, ...],
    dealer_hits_soft_17: bool,
) -> float:
    player_total, _ = total_and_soft_from_cards(cards)
    if player_total > 21:
        return -1.0

    ev = 0.0
    for dealer_outcome, probability in dealer_distribution_from_upcard(
        dealer_card,
        deck_counts,
        dealer_hits_soft_17,
    ):
        ev += probability * compare_player_to_dealer(player_total, dealer_outcome)
    return ev


@lru_cache(maxsize=None)
def hit_ev(
    cards: Tuple[str, ...],
    dealer_card: str,
    deck_counts: Tuple[int, ...],
    dealer_hits_soft_17: bool,
) -> float:
    ev = 0.0

    for card, probability, next_counts in draw_probabilities(deck_counts):
        next_cards = cards + (card,)
        total, _ = total_and_soft_from_cards(next_cards)
        if total > 21:
            next_ev = -1.0
        else:
            # After taking a normal hit, the remaining player choices are hit or stand.
            next_ev = max(
                stand_ev(next_cards, dealer_card, next_counts, dealer_hits_soft_17),
                hit_ev(next_cards, dealer_card, next_counts, dealer_hits_soft_17),
            )
        ev += probability * next_ev

    return ev


@lru_cache(maxsize=None)
def double_ev(
    cards: Tuple[str, ...],
    dealer_card: str,
    deck_counts: Tuple[int, ...],
    dealer_hits_soft_17: bool,
) -> float:
    ev = 0.0

    for card, probability, next_counts in draw_probabilities(deck_counts):
        next_cards = cards + (card,)
        ev += probability * 2.0 * stand_ev(
            next_cards,
            dealer_card,
            next_counts,
            dealer_hits_soft_17,
        )

    return ev


@lru_cache(maxsize=None)
def split_ev(
    pair_card: str,
    dealer_card: str,
    deck_counts: Tuple[int, ...],
    dealer_hits_soft_17: bool,
    can_double_after_split: bool,
) -> float:
    """
    Approximate split EV as two sequential split hands with no resplitting.
    Split aces receive one card only, matching many common casino rules.
    """
    pair_card = normalize_card(pair_card)
    ev = 0.0

    for first_draw, first_probability, after_first in draw_probabilities(deck_counts):
        first_hand = (pair_card, first_draw)
        first_ev = stand_ev(first_hand, dealer_card, after_first, dealer_hits_soft_17)
        if pair_card != "A":
            first_ev = max(
                first_ev,
                hit_ev(first_hand, dealer_card, after_first, dealer_hits_soft_17),
            )
            if can_double_after_split:
                first_ev = max(
                    first_ev,
                    double_ev(first_hand, dealer_card, after_first, dealer_hits_soft_17),
                )

        second_ev_total = 0.0
        for second_draw, second_probability, after_second in draw_probabilities(after_first):
            second_hand = (pair_card, second_draw)
            second_ev = stand_ev(second_hand, dealer_card, after_second, dealer_hits_soft_17)
            if pair_card != "A":
                second_ev = max(
                    second_ev,
                    hit_ev(second_hand, dealer_card, after_second, dealer_hits_soft_17),
                )
                if can_double_after_split:
                    second_ev = max(
                        second_ev,
                        double_ev(second_hand, dealer_card, after_second, dealer_hits_soft_17),
                    )
            second_ev_total += second_probability * second_ev

        ev += first_probability * (first_ev + second_ev_total)

    return ev


def natural_blackjack_ev(state: GameState, deck_counts: Tuple[int, ...]) -> float:
    """
    Natural blackjack pays 3:2 unless the dealer also has a natural blackjack.
    """
    dealer_card = normalize_card(state.dealer_card)
    needed_values = {"A"} if card_value(dealer_card) == 10 else {"10", "J", "Q", "K"}
    if dealer_card == "A":
        needed_values = {"10", "J", "Q", "K"}
    elif card_value(dealer_card) != 10:
        return 1.5

    total_cards = sum(deck_counts)
    if total_cards <= 0:
        return 1.5

    dealer_blackjack_probability = sum(
        deck_counts[RANKS.index(rank)] for rank in needed_values
    ) / total_cards
    return (dealer_blackjack_probability * 0.0) + ((1 - dealer_blackjack_probability) * 1.5)


def calculate_action_evs(state: GameState, hand_info: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate EVs in units of the original bet.

    +1.00 means winning one original bet on average.
    -1.00 means losing one original bet on average.
    Double and split can naturally exceed +/-1 because more money is at risk.
    """
    deck_counts = fresh_deck_counts(state.deck_count)
    deck_counts = remove_visible_cards(deck_counts, state.player_cards + [state.dealer_card])
    player_cards = tuple(state.player_cards)

    if hand_info["is_blackjack"]:
        return {"Stand": natural_blackjack_ev(state, deck_counts)}

    if hand_info["is_bust"]:
        return {"Bust": -1.0}

    evs = {
        "Stand": stand_ev(player_cards, state.dealer_card, deck_counts, state.dealer_hits_soft_17),
        "Hit": hit_ev(player_cards, state.dealer_card, deck_counts, state.dealer_hits_soft_17),
    }

    if state.can_double and len(state.player_cards) == 2:
        evs["Double"] = double_ev(
            player_cards,
            state.dealer_card,
            deck_counts,
            state.dealer_hits_soft_17,
        )

    if state.can_split and len(state.player_cards) == 2 and hand_info["is_pair"]:
        evs["Split"] = split_ev(
            state.player_cards[0],
            state.dealer_card,
            deck_counts,
            state.dealer_hits_soft_17,
            True,
        )

    return evs


def best_ev_action(action_evs: Dict[str, float]) -> str:
    return max(action_evs, key=action_evs.get)


def ev_confidence(action_evs: Dict[str, float], action: str) -> float:
    """
    Convert EV separation into a readable confidence score for the UI.
    """
    if len(action_evs) <= 1:
        return 0.99

    sorted_evs = sorted(action_evs.values(), reverse=True)
    margin = sorted_evs[0] - sorted_evs[1]
    return min(0.99, max(0.55, 0.58 + margin * 2.2))


def format_action_evs(action_evs: Dict[str, float]) -> Dict[str, float]:
    return {action: round(ev, 4) for action, ev in sorted(action_evs.items())}


# --------------------------------------------------
# Explanation helpers
# --------------------------------------------------
def classify_hand(cards: List[str]) -> Dict[str, Any]:
    total = hand_value(cards)
    soft = is_soft_hand(cards)
    pair = is_pair(cards)
    blackjack = is_blackjack(cards)

    return {
        "cards": [normalize_card(c) for c in cards],
        "total": total,
        "is_soft": soft,
        "is_pair": pair,
        "is_blackjack": blackjack,
        "is_bust": is_bust(cards),
    }


def dealer_pressure_label(dealer: int) -> str:
    if dealer in [2, 3, 4, 5, 6]:
        return "weak"
    if dealer in [7, 8, 9]:
        return "neutral"
    return "strong"


def hand_descriptor(hand_info: Dict[str, Any]) -> str:
    total = hand_info["total"]
    if hand_info["is_blackjack"]:
        return "natural blackjack"
    if hand_info["is_pair"]:
        pair_rank = normalize_card(hand_info["cards"][0])
        if pair_rank == "A":
            return "pair of aces"
        if card_value(pair_rank) == 10:
            return "pair of 10-value cards"
        return f"pair of {pair_rank}s"
    if hand_info["is_soft"]:
        return f"soft {total}"
    return f"hard {total}"


def action_reason(state: GameState, action: str, hand_info: Dict[str, Any]) -> str:
    dealer = dealer_upcard_value(state.dealer_card)
    total = hand_info["total"]

    if hand_info["is_blackjack"]:
        return "This is a natural blackjack, which is already the strongest starting hand."

    if hand_info["is_pair"]:
        pair_rank = normalize_card(state.player_cards[0])
        if pair_rank == "A":
            return "Splitting aces gives you a chance to build two stronger hands instead of keeping a single soft 12."
        if card_value(pair_rank) == 8:
            return "Splitting 8s breaks up a weak hard 16, which is usually one of the worst totals to keep together."
        if action == "Split":
            return f"Pair strategy favors splitting here against a dealer {state.dealer_card} to improve long-run outcomes."
        if action == "Stand":
            return f"Keeping this pair together is stronger than splitting it against a dealer {state.dealer_card}."
        if action == "Hit":
            return f"This pair is usually played as a regular hand against a dealer {state.dealer_card}, and hitting gives the better long-run result."

    if hand_info["is_soft"]:
        if action == "Double":
            return f"A soft {total} against a dealer {state.dealer_card} is a strong doubling spot because you can improve without busting on one extra card."
        if action == "Stand":
            return f"A soft {total} is already strong enough to stand here against a dealer {state.dealer_card}."
        return f"A soft {total} against a dealer {state.dealer_card} often benefits from taking another card because the ace gives extra flexibility."

    if action == "Stand":
        return f"A hard {total} against a dealer {state.dealer_card} is usually strong enough, or the dealer is weak enough, that standing is preferred."
    if action == "Double":
        return f"A hard {total} against a dealer {state.dealer_card} is a favorable doubling opportunity because the starting total is strong."
    if action == "Hit":
        return f"A hard {total} against a dealer {state.dealer_card} is usually too weak to stand on, so hitting is preferred."

    return "This action is recommended by the blackjack strategy engine."


def common_mistake_note(state: GameState, action: str, hand_info: Dict[str, Any]) -> str:
    dealer = dealer_upcard_value(state.dealer_card)
    total = hand_info["total"]

    if hand_info["is_blackjack"]:
        return "Do not overthink a natural blackjack. You already started with the best possible two-card hand."

    if hand_info["is_pair"]:
        pair_rank = normalize_card(state.player_cards[0])
        if card_value(pair_rank) == 8:
            return "Many players hate breaking up 8,8 because splitting creates two uncertain hands, but keeping hard 16 together is usually worse."
        if pair_rank == "A":
            return "A common leak is treating aces like a made 12. Splitting them gives each ace a chance to become a much stronger hand."
        if card_value(pair_rank) == 10 and action == "Stand":
            return "Some players get fancy with 10-value pairs, but 20 is already so strong that splitting usually gives away value."

    if hand_info["is_soft"] and action == "Hit":
        return "Soft hands look safer than they are strong. The ace protects you from busting on one card, so taking another card is often correct."

    if hand_info["is_soft"] and action == "Double":
        return "This can feel aggressive, but soft doubles work because you cannot bust on the next card and the dealer is in a vulnerable spot."

    if not hand_info["is_soft"] and total == 16 and action == "Hit":
        return "Hard 16 is one of the most uncomfortable hands in blackjack, so standing here is a very common mistake."

    if not hand_info["is_soft"] and total == 12 and action == "Stand":
        return "Standing on 12 can feel passive, but against the right dealer upcards you would rather let the dealer take the risk."

    if action == "Double":
        return "The mistake here is playing too small. Your total is strong enough that winning more when the dealer is weak is worth it."

    if action == "Hit":
        return "The usual mistake is respecting your total too much. In this matchup, standing loses more often over the long run."

    if action == "Stand":
        return "The common mistake is taking an unnecessary hit. Here, your hand is already doing its job or the dealer is the one under pressure."

    return "The main coaching idea is to follow the matchup, not just how scary your total feels."


def teaching_tip(state: GameState, action: str, hand_info: Dict[str, Any]) -> str:
    dealer = dealer_upcard_value(state.dealer_card)
    total = hand_info["total"]

    if hand_info["is_blackjack"]:
        return "Teaching tip: when you start with 21 in two cards, the hand is finished from a strategy perspective."

    if hand_info["is_pair"] and action == "Split":
        return "Teaching tip: pair strategy is about whether the two cards are stronger together or as two new starting hands."

    if hand_info["is_soft"]:
        if action in ["Hit", "Double"]:
            return "Teaching tip: soft hands have built-in flexibility because the ace can drop from 11 to 1."
        return "Teaching tip: once a soft hand is already strong enough, you stop pressing and let the dealer act first."

    if action == "Stand" and dealer in [4, 5, 6]:
        return "Teaching tip: against weak dealer cards, you often win by avoiding a bust and forcing the dealer to draw."

    if action == "Hit" and total >= 12:
        return "Teaching tip: with stiff hands like 12 through 16, the right play depends heavily on the dealer upcard."

    if action == "Double":
        return "Teaching tip: doubling is strongest when you are likely ahead now and only need one good card to press the edge."

    return "Teaching tip: basic strategy is matchup-based, so the same total can change actions against different dealer upcards."


def ev_summary(action_evs: Dict[str, float], action: str) -> str:
    if not action_evs or action not in action_evs:
        return ""

    ordered = sorted(action_evs.items(), key=lambda item: item[1], reverse=True)
    best_ev = action_evs[action]
    summary = f"The EV model rates {action} at {best_ev:+.3f} units per original bet."

    if len(ordered) > 1:
        second_action, second_ev = ordered[1]
        summary += f" The next-best option is {second_action} at {second_ev:+.3f}, a margin of {best_ev - second_ev:+.3f}."

    return summary


def math_reason(
    state: GameState,
    action: str,
    hand_info: Dict[str, Any],
    action_evs: Dict[str, float] | None = None,
) -> str:
    dealer = dealer_upcard_value(state.dealer_card)
    total = hand_info["total"]
    dealer_pressure = dealer_pressure_label(dealer)
    ev_text = ev_summary(action_evs or {}, action)

    if hand_info["is_blackjack"]:
        return f"Math view: a two-card 21 is already the top starting outcome, so there is no higher-EV action to consider. {ev_text}".strip()

    if hand_info["is_pair"]:
        pair_rank = normalize_card(state.player_cards[0])
        if pair_rank == "A":
            return f"Math view: splitting aces increases the chance of turning one weak soft 12 into two hands that can each reach 18 to 21. {ev_text}".strip()
        if card_value(pair_rank) == 8:
            return f"Math view: splitting 8,8 avoids locking in a hard 16, one of the weakest non-bust totals in blackjack. {ev_text}".strip()
        if card_value(pair_rank) == 10 and action == "Stand":
            return f"Math view: a made 20 already wins so often that splitting usually lowers your expected value. {ev_text}".strip()
        if action == "Split":
            return f"Math view: the EV model prefers splitting because two fresh hands outperform keeping this pair together against a dealer {state.dealer_card}. {ev_text}".strip()

    if hand_info["is_soft"] and action == "Double":
        return f"Math view: soft {total} can improve on one card without immediate bust risk, and dealer {state.dealer_card} is weak enough to justify pressing the edge. {ev_text}".strip()

    if hand_info["is_soft"] and action == "Hit":
        return f"Math view: soft {total} is not strong enough yet, and the ace lowers the penalty of taking one more card. {ev_text}".strip()

    if hand_info["is_soft"] and action == "Stand":
        return f"Math view: soft {total} already performs well enough that the extra volatility from hitting is not worth it. {ev_text}".strip()

    if action == "Stand":
        return f"Math view: with {hand_descriptor(hand_info)}, your bust risk from hitting is too costly, especially against a {dealer_pressure} dealer card like {state.dealer_card}. {ev_text}".strip()

    if action == "Double":
        return f"Math view: {hand_descriptor(hand_info)} is strong enough that increasing the bet has better long-run value than taking a normal hit. {ev_text}".strip()

    if action == "Hit":
        return f"Math view: standing on {hand_descriptor(hand_info)} leaves you behind too often against dealer {state.dealer_card}, so the lower-loss play is to draw. {ev_text}".strip()

    return f"Math view: this play follows the best long-run expectation in the current matchup. {ev_text}".strip()


def build_coaching_payload(
    state: GameState,
    action: str,
    hand_info: Dict[str, Any],
    action_evs: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    primary_reason = action_reason(state, action, hand_info)
    mistake_note = common_mistake_note(state, action, hand_info)
    tip = teaching_tip(state, action, hand_info)
    math_text = math_reason(state, action, hand_info, action_evs)
    dealer = dealer_upcard_value(state.dealer_card)

    ev_text = ev_summary(action_evs or {}, action)
    beginner = f"{action}, because {primary_reason[0].lower() + primary_reason[1:]} {ev_text} {mistake_note}".strip()
    quick = f"{action} because {hand_descriptor(hand_info)} plays best against dealer {state.dealer_card} here."

    if hand_info["is_blackjack"]:
        quick = "Stand because a natural blackjack is already complete."
    elif action == "Hit" and hand_info["total"] == 16 and dealer >= 7:
        quick = f"Hit because hard 16 stands lose too often against a dealer {state.dealer_card}."
    elif action == "Stand" and hand_info["total"] in [12, 13, 14, 15, 16] and dealer in [4, 5, 6]:
        quick = f"Stand because the dealer is more likely to self-destruct from {state.dealer_card} than you are to improve safely."

    return {
        "recommended_action": action,
        "quick": quick,
        "beginner": beginner,
        "math": math_text,
        "common_mistake": mistake_note,
        "teaching_tip": tip,
        "decision_summary": f"{primary_reason} {ev_text}".strip(),
    }


def simple_confidence(state: GameState, action: str, hand_info: Dict[str, Any]) -> float:
    """
    Lightweight confidence score for MVP UI.
    Not a true probability.
    """
    dealer = dealer_upcard_value(state.dealer_card)
    total = hand_info["total"]

    if hand_info["is_blackjack"]:
        return 0.99

    if action == "Split" and hand_info["is_pair"]:
        rank = normalize_card(state.player_cards[0])
        if rank == "A" or card_value(rank) == 8:
            return 0.96
        return 0.85

    if action == "Double":
        if total in [10, 11]:
            return 0.9
        if hand_info["is_soft"] and total in [17, 18]:
            return 0.84
        return 0.8

    if action == "Stand":
        if total >= 17:
            return 0.94
        if total in [12, 13, 14, 15, 16] and dealer in [4, 5, 6]:
            return 0.86
        return 0.78

    if action == "Hit":
        if total <= 11:
            return 0.95
        if total in [12, 13, 14, 15, 16] and dealer in [7, 8, 9, 10, 11]:
            return 0.88
        return 0.8

    return 0.75


# --------------------------------------------------
# Main engine
# --------------------------------------------------
def best_action(state: GameState) -> Dict[str, Any]:
    """
    Main recommendation function.
    Returns a structured dict that can be fed into your app or GPT layer.
    """
    state = state.normalized()
    hand_info = classify_hand(state.player_cards)
    action_evs = calculate_action_evs(state, hand_info)

    # Immediate terminal cases
    if hand_info["is_blackjack"]:
        action = "Stand"
        explanation = action_reason(state, action, hand_info)
        coaching = build_coaching_payload(state, action, hand_info, action_evs)
        return {
            "state": asdict(state),
            "hand_info": hand_info,
            "recommended_action": action,
            "confidence": ev_confidence(action_evs, action),
            "best_ev": round(action_evs[action], 4),
            "ev_margin": None,
            "action_evs": format_action_evs(action_evs),
            "model_type": "finite_deck_expected_value",
            "explanation": explanation,
            "coach": coaching,
        }

    if hand_info["is_bust"]:
        return {
            "state": asdict(state),
            "hand_info": hand_info,
            "recommended_action": "Bust",
            "confidence": 1.0,
            "best_ev": -1.0,
            "ev_margin": None,
            "action_evs": {"Bust": -1.0},
            "model_type": "finite_deck_expected_value",
            "explanation": "The hand total is over 21, so the hand is already bust.",
            "coach": {
                "recommended_action": "Bust",
                "quick": "Bust because the total is already over 21.",
                "beginner": "Bust, because your hand is already over 21 and the decision is no longer live.",
                "math": "Math view: once the hand is over 21, the outcome is already locked in as a loss.",
                "common_mistake": "The coaching move here is not strategic; it is just recognizing that the hand is already dead.",
                "teaching_tip": "Teaching tip: count the total before worrying about advanced strategy.",
                "decision_summary": "The hand total is over 21, so the hand is already bust.",
            },
        }

    action = best_ev_action(action_evs)
    explanation = action_reason(state, action, hand_info)
    coaching = build_coaching_payload(state, action, hand_info, action_evs)
    ordered_evs = sorted(action_evs.values(), reverse=True)
    ev_margin = ordered_evs[0] - ordered_evs[1] if len(ordered_evs) > 1 else None

    return {
        "state": asdict(state),
        "hand_info": hand_info,
        "recommended_action": action,
        "confidence": ev_confidence(action_evs, action),
        "best_ev": round(action_evs[action], 4),
        "ev_margin": round(ev_margin, 4) if ev_margin is not None else None,
        "action_evs": format_action_evs(action_evs),
        "model_type": "finite_deck_expected_value",
        "explanation": explanation,
        "coach": coaching,
    }


# --------------------------------------------------
# Convenience API for UI / app
# --------------------------------------------------
def recommend_action(
    player_cards: List[str],
    dealer_card: str,
    can_double: bool = True,
    can_split: bool = True,
    dealer_hits_soft_17: bool = False,
    deck_count: int = 2,
) -> Dict[str, Any]:
    """
    Simple wrapper for app code.
    """
    state = GameState(
        player_cards=player_cards,
        dealer_card=dealer_card,
        can_double=can_double,
        can_split=can_split,
        dealer_hits_soft_17=dealer_hits_soft_17,
        deck_count=deck_count,
    )
    return best_action(state)


def format_for_gpt(result: Dict[str, Any]) -> str:
    """
    Converts engine output into a compact text prompt for your nanoGPT model.
    """
    state = result["state"]
    hand = result["hand_info"]
    return (
        f"Player: {','.join(state['player_cards'])} | "
        f"Dealer: {state['dealer_card']} | "
        f"Total: {hand['total']} | "
        f"Soft: {hand['is_soft']} | "
        f"Pair: {hand['is_pair']} | "
        f"Action: {result['recommended_action']} | "
        f"Best EV: {result.get('best_ev')} | "
        f"EVs: {result.get('action_evs', {})} | "
        f"Reason: {result['explanation']}"
    )


# --------------------------------------------------
# Quick test
# --------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        (["8", "8"], "6"),
        (["10", "6"], "7"),
        (["A", "7"], "9"),
        (["5", "5"], "6"),
        (["9", "9"], "7"),
        (["10", "2"], "4"),
        (["A", "6"], "3"),
    ]

    for player_cards, dealer_card in test_cases:
        result = recommend_action(player_cards, dealer_card)
        print("=" * 70)
        print(f"Player: {player_cards} | Dealer: {dealer_card}")
        print(f"Recommended Action: {result['recommended_action']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Explanation: {result['explanation']}")
        print("GPT Prompt Format:")
        print(format_for_gpt(result))
