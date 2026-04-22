# Session Checkpoint

Saved state for continuing work on the Blackjack AI Coach MVP.

## Current Demo Folder

```text
C:\Users\Ivan\OneDrive\Documents\BJ Coach 301\Project-301--Implemented-BJ-Coach\mvp
```

## Run The Demo

Open PowerShell and run:

```powershell
cd "C:\Users\Ivan\OneDrive\Documents\BJ Coach 301\Project-301--Implemented-BJ-Coach\mvp"
& "C:\Users\Ivan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m streamlit run app.py
```

Then open:

```text
http://127.0.0.1:8501
```

Keep the PowerShell window open while presenting or editing. If it closes, the demo server stops.

## Current Product State

- Hero: Blackjack AI Coach with the tagline "Table Sense, Not Just Table Math"
- Membership tiers:
  - Table Coach - Free
  - EV Edge - $19 / month
  - Bankroll Desk - $39 / month
- Coach modes:
  - Table Coach
  - EV Edge
  - Bankroll Desk
- Membership Tier and Coach Mode are synchronized.
- Free tier shows classical coaching.
- EV Edge shows expected value outputs.
- Bankroll Desk adds bankroll/risk framing.
- Green gradient styling is applied across dropdowns, expanders, copy blocks, and output areas.

## Main Files To Edit

```text
app.py
blackjack_engine.py
PRESENTATION_GUIDE.md
README.md
requirements.txt
.streamlit/config.toml
```

## Latest GitHub Prep Folder

```text
C:\Users\Ivan\OneDrive\Documents\BJ Coach 301\Project-301--Implemented-BJ-Coach\github_upload
```

If more UI or code changes are made, refresh the `github_upload` folder before uploading to GitHub.

