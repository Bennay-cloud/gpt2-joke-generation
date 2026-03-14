import json
import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "model"
TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"
EVAL_RESULT_PATH = OUTPUT_DIR / "evaluation_result.json"
PROMOTION_DIR = OUTPUT_DIR / "promotion"

VERSION_FILE = PROMOTION_DIR / "version.json"
METADATA_FILE = PROMOTION_DIR / "metadata.json"
LATEST_FILE = PROMOTION_DIR / "latest.json"


def ensure_model_ready():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing model directory: {MODEL_DIR}")
    if not TOKENIZER_DIR.exists():
        raise FileNotFoundError(f"Missing tokenizer directory: {TOKENIZER_DIR}")
    if not EVAL_RESULT_PATH.exists():
        raise FileNotFoundError(f"Missing evaluation result: {EVAL_RESULT_PATH}")


def load_eval_result():
    with open(EVAL_RESULT_PATH, "r", encoding="utf-8") as f:
        result = json.load(f)

    if not result.get("approved", False):
        raise ValueError("Model is not approved for promotion.")

    return result


def get_next_version():
    PROMOTION_DIR.mkdir(parents=True, exist_ok=True)

    if VERSION_FILE.exists():
        with open(VERSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        current = int(data.get("last_version", 0))
    else:
        current = 0

    next_version = current + 1

    with open(VERSION_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_version": next_version}, f, indent=2)

    return f"v{next_version}"


def write_metadata(version, eval_result):
    metadata = {
        "version": version,
        "approved": True,
        "eval_loss": eval_result["eval_loss"],
        "threshold": eval_result["threshold"],
        "model_path": str(MODEL_DIR),
        "tokenizer_path": str(TOKENIZER_DIR),
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(LATEST_FILE, "w", encoding="utf-8") as f:
        json.dump({"latest_version": version}, f, indent=2)

    return metadata


def prepare_local_version_package(version):
    version_dir = PROMOTION_DIR / version

    if version_dir.exists():
        shutil.rmtree(version_dir)

    (version_dir / "model").mkdir(parents=True, exist_ok=True)
    (version_dir / "tokenizer").mkdir(parents=True, exist_ok=True)

    shutil.copytree(MODEL_DIR, version_dir / "model", dirs_exist_ok=True)
    shutil.copytree(TOKENIZER_DIR, version_dir / "tokenizer", dirs_exist_ok=True)
    shutil.copy2(METADATA_FILE, version_dir / "metadata.json")

    return version_dir


def main():
    ensure_model_ready()
    eval_result = load_eval_result()
    version = get_next_version()
    metadata = write_metadata(version, eval_result)
    version_dir = prepare_local_version_package(version)

    print("Model promoted successfully.")
    print(json.dumps(metadata, indent=2))
    print(f"Local version package ready at: {version_dir}")


if __name__ == "__main__":
    main()
