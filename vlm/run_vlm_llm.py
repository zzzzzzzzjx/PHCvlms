#!/usr/bin/env python3
"""
Call a vision-language model with a fixed base prompt + user prompt, then
export body-part weights to a CSV compatible with humanoid_im VLM loading.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm.llm_client import (  # noqa: E402
    build_prompt,
    call_vlm,
    decode_motion_weights,
    load_llm_config,
    weights_to_csv_rows,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send prompt + frame images to a VLM and save weights CSV."
    )
    parser.add_argument(
        "--config",
        default="vlm/cfg/llm_config.example.yaml",
        help="YAML with base_url/api_key/model.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Paths to consecutive frame images (left-to-right = time order).",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional extra instruction appended after the built-in base prompt.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Destination CSV path (frame,waist,left_hand,right_hand,left_foot,right_foot).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name from config.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Frame count for CSV; defaults to len(images).",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save parsed JSON weights for debugging.",
    )
    parser.add_argument(
        "--save-raw",
        default=None,
        help="Optional path to save full raw response JSON from the LLM API.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if len(args.images) != 1:
            raise ValueError("This script expects a single pre-stitched image (e.g., 15 frames combined). Please pass exactly one image path.")

        cfg = load_llm_config(args.config)
        if args.model:
            cfg["model"] = args.model

        prompt = build_prompt(args.prompt)
        message_content, full_response = call_vlm(prompt, args.images, cfg)
        motion_weights = decode_motion_weights(message_content)
        frame_count = args.frames if args.frames is not None else len(args.images)
        csv_rows = weights_to_csv_rows(motion_weights, frame_count)
        write_csv(csv_rows, args.output_csv)

        # 终端额外输出动作分析与权重理由（如果模型提供）
        analysis = motion_weights.get("analysis_en", "").strip()
        rationale = motion_weights.get("rationale_en", "").strip()
        if analysis:
            print("\n[Action Analysis]\n" + analysis)
        if rationale:
            print("\n[Weight Rationale]\n" + rationale)

        if args.save_json:
            Path(args.save_json).write_text(
                json.dumps(motion_weights, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if args.save_raw:
            Path(args.save_raw).write_text(
                json.dumps(full_response, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        print(f"✓ Saved weights CSV to {args.output_csv}")
        if args.save_json:
            print(f"  Parsed JSON saved to {args.save_json}")
        if args.save_raw:
            print(f"  Raw response saved to {args.save_raw}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Failed to generate VLM weights: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
