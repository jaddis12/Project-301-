# Blackjack AI Coach MVP Report

## Overview

The Blackjack AI Coach is a Streamlit MVP that recommends blackjack actions and explains them in plain language. The core recommendation comes from a rules engine, while the project also includes a nanoGPT-style training and inference workflow for generated explanations.

## User Flow

1. The user opens the app with `streamlit run app.py`.
2. The user enters two player cards and the dealer upcard.
3. The user optionally enters extra player cards after a hit.
4. The user chooses rule options in the sidebar.
5. The app returns a recommended move, confidence score, player total, and explanation.

## Technical Approach

The project uses a hybrid AI architecture:

- Deterministic logic handles blackjack strategy decisions.
- A GPT-style model is trained on blackjack explanation text.
- The app combines both into a coaching-focused user experience.

This avoids asking the language model to perform exact game logic while still satisfying the model training and inference requirements.

## Deliverables

| Requirement | Status | Evidence |
| --- | --- | --- |
| Working `/mvp/` directory exists | Complete | This folder contains the MVP files |
| Training code or notebook included | Complete | `train.py`, `model.py`, `build_blackjack_dataset.py` |
| Inference code or notebook included | Complete | `sample.py` |
| Demo entrypoint included | Complete | `app.py` |
| `README.md` completed | Complete | `README.md` |
| `report.md` completed | Complete | `report.md` |

## Limitations

The rules engine is suitable for an MVP and class demonstration, but it is not a full casino-grade expected-value simulator. The nanoGPT-style model is intended for explanation generation, not for deciding the mathematically best move.

## Conclusion

The MVP is complete and organized for submission. It includes a runnable demo, training workflow, inference script, README, and project report.

