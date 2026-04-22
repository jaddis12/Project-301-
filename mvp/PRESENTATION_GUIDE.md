# Blackjack AI Coach MVP Presentation Guide

## Demo Name

**Blackjack AI Coach**

Tagline:

**Table Sense, Not Just Table Math**

One-sentence pitch:

Blackjack AI Coach is a Streamlit MVP that gives blackjack training recommendations, compares long-run expected value for legal moves, and turns the result into coaching language a player can remember next time.

## Streamlit Cloud Deploy

After uploading the current `mvp/` folder to GitHub, deploy with:

```text
Main file path: mvp/app.py
```

The app depends on:

```text
mvp/requirements.txt
```

The Streamlit config file is:

```text
mvp/.streamlit/config.toml
```

## What To Show

Run the demo from the `mvp` folder:

```powershell
cd "C:\Users\Ivan\OneDrive\Documents\BJ Coach 301\Project-301--Implemented-BJ-Coach\mvp"
python -m streamlit run app.py
```

If Python is not on PATH, use the bundled runtime already used for this workspace:

```powershell
cd "C:\Users\Ivan\OneDrive\Documents\BJ Coach 301\Project-301--Implemented-BJ-Coach\mvp"
& "C:\Users\Ivan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m streamlit run app.py
```

Open:

```text
http://127.0.0.1:8501
```

## Required Dependencies

The dependency file is:

```text
requirements.txt
```

Current dependencies:

```text
streamlit
torch
matplotlib
```

Install with:

```powershell
pip install -r requirements.txt
```

or with the bundled runtime:

```powershell
& "C:\Users\Ivan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m pip install -r requirements.txt
```

## Streamlit Configuration

The demo configuration lives at:

```text
.streamlit/config.toml
```

It pins the local demo to:

```text
address = 127.0.0.1
port = 8501
```

It also disables Streamlit usage stats and sets the base theme colors.

## Main Code Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit user interface and tiered product demo |
| `blackjack_engine.py` | Blackjack hand logic and finite-deck expected-value model |
| `model.py` | nanoGPT-style model definition |
| `train.py` | Character-level model training script |
| `sample.py` | Inference script for a trained checkpoint |
| `build_blackjack_dataset.py` | Synthetic blackjack training data generator |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Local Streamlit demo configuration |

## Product Tiers

The app presents three model tiers from the `Membership Tier` dropdown.

### Table Coach - Free

Model:

```text
Classical Coach Model
```

Output:

- Best move
- Player total
- Plain-English coaching
- Strategic pattern to remember
- Upgrade preview instead of full EV table
- Coach mode syncs to the free tier language

### EV Edge - $19 / month

Model:

```text
Expected Value Model
```

Output:

- Best move
- Expected value
- EV margin
- Action-by-action EV comparison
- Long-run financial explanation
- Coach mode syncs to EV Edge automatically

### Bankroll Desk - $39 / month

Model:

```text
Bankroll-Aware EV Model
```

Output:

- Everything in EV Edge
- Bankroll and current bet inputs
- Short bankroll-aware coach call
- Medium-length bankroll lens
- Full 1-2 paragraph bankroll desk analysis
- Coach mode syncs to Bankroll Desk automatically

## Recommended Demo Script

1. Open the app and point out the tagline: **Table Sense, Not Just Table Math**.
2. Explain that the app is not trying to be a gambling shortcut. It is framed as a training assistant.
3. Show the `Membership Tier` dropdown.
4. Start with `Table Coach - Free`.
5. Enter this test hand:

```text
Player: 8,8
Dealer: 6
```

6. Click `Analyze Hand`.
7. Explain that the free tier gives an understandable coaching answer.
8. Switch to `EV Edge - $19 / month`.
9. Re-run or review the output and open `Show EV comparison`.
10. Explain that this tier exposes the model's expected value comparison.
11. Switch to `Bankroll Desk - $39 / month`.
12. Show the bankroll/risk framing.
13. Close by explaining the architecture: deterministic expected-value logic for decisions, language model workflow for natural-language explanation.

## Good Test Hands

Use these during the presentation:

```text
Player: 8,8 | Dealer: 6
Player: 10,6 | Dealer: 7
Player: A,7 | Dealer: 9
Player: 5,5 | Dealer: 6
```

## Architecture

```text
User cards
  -> blackjack_engine.py
  -> finite-deck EV comparison
  -> recommended action
  -> tiered output layer in app.py
  -> coaching explanation
  -> optional nanoGPT explanation if checkpoint exists
```

## Expected-Value Model

The EV model in `blackjack_engine.py` estimates legal actions in units of the original bet.

Examples:

```text
+1.000 means winning one original bet on average.
 0.000 means break-even on average.
-1.000 means losing one original bet on average.
```

Double and split can move beyond the usual one-unit scale because those actions put more money at risk.

## AI / ML Requirement

This project includes a nanoGPT-style workflow:

1. Build training data:

```powershell
python build_blackjack_dataset.py
```

2. Train the model:

```powershell
python train.py
```

3. Sample from the trained model:

```powershell
python sample.py
```

The trained checkpoint path is:

```text
out/best_model.pt
```

The tokenizer metadata path is:

```text
out/meta.json
```

The Streamlit app still works if no checkpoint is available. In that case, it uses the EV-based coach explanation.

## Team Talking Points

- The MVP separates decision quality from explanation quality.
- The EV engine makes the move recommendation.
- The language layer makes the recommendation understandable.
- The product tiering creates a natural business model:
  - free coaching,
  - paid EV transparency,
  - premium bankroll-aware decision support.
- The design is meant for education and practice.

## Current Local URL

When running locally:

```text
http://127.0.0.1:8501
```
