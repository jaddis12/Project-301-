# sample.py
import json
from pathlib import Path
import torch
from model import GPT, GPTConfig

SCRIPT_DIR = Path(__file__).resolve().parent
checkpoint_path = SCRIPT_DIR / "out" / "best_model.pt"
meta_path = SCRIPT_DIR / "out" / "meta.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

stoi = meta["stoi"]
itos = {int(k): v for k, v in meta["itos"].items()}

def encode(s: str):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

checkpoint = torch.load(checkpoint_path, map_location=device)
config = GPTConfig(**checkpoint["model_args"])
model = GPT(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prompt = (
    "Player: 8,8 | Dealer: 6 | Total: 16 | Soft: False | Pair: True | "
    "Double Allowed: True | Split Allowed: True | Dealer Hits Soft 17: False | Deck Count: 2 | "
    "Action: Split | Best EV: 0.3328 | EVs: {'Double': -0.8286, 'Hit': -0.4143, 'Split': 0.3328, 'Stand': -0.1654} | "
    "Reason: Splitting 8s breaks up a weak hard 16, which is usually one of the worst totals to keep together. | "
    "Tier: EV Edge | Voice: expected_value | Coach Call:"
)
x = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

with torch.no_grad():
    y = model.generate(x, max_new_tokens=120, temperature=0.8, top_k=30)

print(decode(y[0].tolist()))
