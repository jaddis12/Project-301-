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
from typing import List, Dict, Any


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


def math_reason(state: GameState, action: str, hand_info: Dict[str, Any]) -> str:
    dealer = dealer_upcard_value(state.dealer_card)
    total = hand_info["total"]
    dealer_pressure = dealer_pressure_label(dealer)

    if hand_info["is_blackjack"]:
        return "Math view: a two-card 21 is already the top starting outcome, so there is no higher-EV action to consider."

    if hand_info["is_pair"]:
        pair_rank = normalize_card(state.player_cards[0])
        if pair_rank == "A":
            return "Math view: splitting aces increases the chance of turning one weak soft 12 into two hands that can each reach 18 to 21."
        if card_value(pair_rank) == 8:
            return "Math view: splitting 8,8 avoids locking in a hard 16, one of the weakest non-bust totals in blackjack."
        if card_value(pair_rank) == 10 and action == "Stand":
            return "Math view: a made 20 already wins so often that splitting usually lowers your expected value."
        if action == "Split":
            return f"Math view: the pair chart prefers splitting because two fresh hands outperform keeping this pair together against a dealer {state.dealer_card}."

    if hand_info["is_soft"] and action == "Double":
        return f"Math view: soft {total} can improve on one card without immediate bust risk, and dealer {state.dealer_card} is weak enough to justify pressing the edge."

    if hand_info["is_soft"] and action == "Hit":
        return f"Math view: soft {total} is not strong enough yet, and the ace lowers the penalty of taking one more card."

    if hand_info["is_soft"] and action == "Stand":
        return f"Math view: soft {total} already performs well enough that the extra volatility from hitting is not worth it."

    if action == "Stand":
        return f"Math view: with {hand_descriptor(hand_info)}, your bust risk from hitting is too costly, especially against a {dealer_pressure} dealer card like {state.dealer_card}."

    if action == "Double":
        return f"Math view: {hand_descriptor(hand_info)} is strong enough that increasing the bet has better long-run value than taking a normal hit."

    if action == "Hit":
        return f"Math view: standing on {hand_descriptor(hand_info)} leaves you behind too often against dealer {state.dealer_card}, so the lower-loss play is to draw."

    return "Math view: this play follows the best long-run expectation in the current matchup."


def build_coaching_payload(state: GameState, action: str, hand_info: Dict[str, Any]) -> Dict[str, Any]:
    primary_reason = action_reason(state, action, hand_info)
    mistake_note = common_mistake_note(state, action, hand_info)
    tip = teaching_tip(state, action, hand_info)
    math_text = math_reason(state, action, hand_info)
    dealer = dealer_upcard_value(state.dealer_card)

    beginner = f"{action}, because {primary_reason[0].lower() + primary_reason[1:]} {mistake_note}"
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
        "decision_summary": primary_reason,
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
    dealer = dealer_upcard_value(state.dealer_card)

    # Immediate terminal cases
    if hand_info["is_blackjack"]:
        action = "Stand"
        explanation = action_reason(state, action, hand_info)
        coaching = build_coaching_payload(state, action, hand_info)
        return {
            "state": asdict(state),
            "hand_info": hand_info,
            "recommended_action": action,
            "confidence": simple_confidence(state, action, hand_info),
            "explanation": explanation,
            "coach": coaching,
        }

    if hand_info["is_bust"]:
        return {
            "state": asdict(state),
            "hand_info": hand_info,
            "recommended_action": "Bust",
            "confidence": 1.0,
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

    # Pair strategy first if exactly two cards and splitting allowed
    if len(state.player_cards) == 2 and hand_info["is_pair"] and state.can_split:
        pair_rank = normalize_card(state.player_cards[0])
        action = pair_action(pair_rank, dealer, can_double=state.can_double)
        explanation = action_reason(state, action, hand_info)
        coaching = build_coaching_payload(state, action, hand_info)
        return {
            "state": asdict(state),
            "hand_info": hand_info,
            "recommended_action": action,
            "confidence": simple_confidence(state, action, hand_info),
            "explanation": explanation,
            "coach": coaching,
        }

    # Soft hand strategy next for exactly two cards containing an ace
    if len(state.player_cards) == 2 and hand_info["is_soft"]:
        action = soft_hand_action(hand_info["total"], dealer, can_double=state.can_double)
        explanation = action_reason(state, action, hand_info)
        coaching = build_coaching_payload(state, action, hand_info)
        return {
            "state": asdict(state),
            "hand_info": hand_info,
            "recommended_action": action,
            "confidence": simple_confidence(state, action, hand_info),
            "explanation": explanation,
            "coach": coaching,
        }

    # Otherwise use hard total strategy
    action = hard_total_action(hand_info["total"], dealer, can_double=state.can_double)
    explanation = action_reason(state, action, hand_info)
    coaching = build_coaching_payload(state, action, hand_info)
    return {
        "state": asdict(state),
        "hand_info": hand_info,
        "recommended_action": action,
        "confidence": simple_confidence(state, action, hand_info),
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
