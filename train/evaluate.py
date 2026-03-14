import json
import os
from pathlib import Path

THRESHOLD = float(os.getenv("EVAL_LOSS_THRESHOLD", 5.0))

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
TRAINER_DIR = OUTPUT_DIR / "trainer"
RESULT_PATH = OUTPUT_DIR / "evaluation_result.json"


def find_latest_trainer_state() -> Path:
    candidates = list(TRAINER_DIR.rglob("trainer_state.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No trainer_state.json found under {TRAINER_DIR}. "
            "Run training first."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_eval_loss(trainer_state_path: Path) -> float:
    with open(trainer_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    eval_losses = [
        entry["eval_loss"]
        for entry in log_history
        if isinstance(entry, dict) and "eval_loss" in entry
    ]

    if not eval_losses:
        raise ValueError("No eval_loss found in trainer_state.json")

    return float(eval_losses[-1])


def main():
    trainer_state_path = find_latest_trainer_state()
    eval_loss = extract_eval_loss(trainer_state_path)

    approved = eval_loss < THRESHOLD

    result = {
        "approved": approved,
        "eval_loss": eval_loss,
        "threshold": THRESHOLD,
        "trainer_state_path": str(trainer_state_path),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

    if not approved:
        raise SystemExit(
            f"Model rejected: eval_loss={eval_loss:.4f} >= threshold={THRESHOLD:.4f}"
        )

    print("Model approved for promotion.")


if __name__ == "__main__":
    main()
