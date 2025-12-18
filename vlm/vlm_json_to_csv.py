#!/usr/bin/env python3
"""
Convert a saved LLM/VLM JSON result into the CSV format expected by humanoid_im.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm.llm_client import decode_motion_weights, weights_to_csv_rows, write_csv  # noqa: E402


def _load_weights(input_path: Path) -> Dict[str, Any]:
    raw = input_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except Exception:
        return decode_motion_weights(raw)

    if isinstance(data, dict) and "body_part_weights" in data:
        return {
            "motion_type": data.get("motion_type", ""),
            "body_part_weights": data["body_part_weights"],
        }

    if isinstance(data, dict):
        try:
            content = data["choices"][0]["message"]["content"]
            return decode_motion_weights(content)
        except Exception:
            pass

    if isinstance(data, str):
        return decode_motion_weights(data)

    raise ValueError("Unsupported JSON structure; expected body_part_weights.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert saved LLM output JSON to a VLM weight CSV."
    )
    parser.add_argument("--input-json", required=True, help="Saved JSON/text file.")
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Destination CSV path (frame,waist,left_hand,right_hand,left_foot,right_foot).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Frame count to replicate weights across (default: 5).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        weights = _load_weights(Path(args.input_json))
        rows = weights_to_csv_rows(weights, args.frames)
        write_csv(rows, args.output_csv)
        print(f"✓ Saved weights CSV to {args.output_csv}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Failed to convert JSON to CSV: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
