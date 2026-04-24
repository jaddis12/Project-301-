# Blackjack AI Coach MVP

Our team propses to create an Ai startup that counts cards at blackjack, and poker tables. 

The Ai will be trained to use basic probability and
expected value method for solving black jack by using
the variable X as all the card outcomes possible while
running it through a probability density formula. The
main tasks that the Ai will tackle are enumerating all
possible future card outcomes, calculating expected
return for each action, and choosing the best one.
The Ai will use EV = P(win)W + P(push)(0) - P(lose)L to
know when to bet more and when the deck favors the
player.

The Ai will also be trained to use expected value and
Baye’s theorem methods for solving poker by using EV =
P(win)(profit) - P(loss)(cost). When using Baye’s
theorem you’ll be observing your opponent’s hand by
using P(hand | Action) = (P(action | hand)P(hand)) /
(P(action)). This helps in poker, because you have to
determine
- What hands they start with
- How likely each hand can produce a win or lose
- Updating each probability after each action

## Files

| File | Blaackjack_app | Purpose |
| --- | --- | --- |
| `app.py` | web UI interface |
| `blackjack_engine.py` | core decision engine |
| `build_blackjack_dataset.py` | generates data for the deck of cards |
| `model.py` | language model & explanation text |
| `train.py` | mearsures & trains losses and wins |
| `sample.py` | demo test file for easier explanations |
| `requirements.txt` | lists of programs needed to run everything |
| `report.md` | short project report explaining MVP |

## Github URL

```bash
 https://github.com/jaddis12/Project-301-/tree/main
```

## Run Demo

```bash
[streamlit run app.py](https://301-project-mp1.streamlit.app/)
```

## Applications Used

```bash
-streamlit
-torch
-Matplotlib
```

## Run Inference

```bash
-Code python
-PyTorch GPT-style model
-HTML/CSS
-TOML
-PowerShell
```
