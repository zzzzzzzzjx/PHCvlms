import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import yaml

# 基础提示词：15 连帧动作分析 + 权重 + 文字解读（英文）
BASE_PROMPT = """You will see a time-ordered strip of 15 consecutive action frames (left to right = earlier to later) covering ~1 second of combat movement. Analyze all 15 frames carefully, assign body-part control weights (1-5), and also provide concise English analysis and rationale.

Body parts: left_arm, right_arm, left_leg, right_leg, torso
Weights: 5=core drive (large/rapid), 4=primary, 3=assist, 2=minor, 1=stationary

Key action types to recognize:
- Combat: right jab/left jab/right hook/left hook/right uppercut/left uppercut/right kick/left kick/knee strike/elbow strike/uppercut/jab/combination punch/defensive block
- Gait: walk/run/jog/sprint/backward/sideways
- Jump: in-place/forward/backward/side
- Other: push/pull/lift/throw/turn/squat/stand

Judgment cues:
- Right straight punch: right_arm=5, torso=4, legs support 2-3
- Left straight punch: left_arm=5, torso=4, legs support 2-3
- Right kick: right_leg=5, torso=4, left_leg support 4
- Left kick: left_leg=5, torso=4, right_leg support 4
- Combo punch: dominant arm=5, torso=4
- Defense: extended arm=4-5

Output JSON only, no extra text:
{
  "motion_type": "short action name (<=10 chars, e.g., right_straight, left_kick, sprint, stand)",
  "body_part_weights": {
    "left_arm": 1-5,
    "right_arm": 1-5,
    "left_leg": 1-5,
    "right_leg": 1-5,
    "torso": 1-5
  },
  "analysis_en": "80-200 words: what happens across the 15 frames (left->right), key poses, strikes, footwork, timing, left/right are accurate.",
  "rationale_en": "80-200 words: why each part got its weight, referring to visible motion magnitude, support vs striking leg/arm, torso rotation, balance."
}

Strict requirements:
1) Respect frame order (left->right) when inferring the action.
2) Action type must be specific and justified by the frames; avoid guessing.
3) Absolutely avoid left/right confusion for arms/legs. Analyze now."""

# 映射到 CSV 列名，保持 humanoid_im.py 期望的格式
BODY_PART_TO_CSV = {
    "torso": "waist",
    "left_arm": "left_hand",
    "right_arm": "right_hand",
    "left_leg": "left_foot",
    "right_leg": "right_foot",
}

CSV_COLUMNS = ["frame", "waist", "left_hand", "right_hand", "left_foot", "right_foot"]


class LlmClientError(Exception):
    """Raised for LLM/VLM pipeline errors."""


def load_llm_config(path: os.PathLike) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    required = ["base_url", "api_key", "model"]
    for key in required:
        if key not in cfg or not cfg[key]:
            raise LlmClientError(f"Missing required config key: {key}")
    return cfg


def build_prompt(user_prompt: str = "") -> str:
    prompt = BASE_PROMPT.strip()
    extra = user_prompt.strip()
    if extra:
        prompt = f"{prompt}\n\n# 补充指令:\n{extra}"
    return prompt


def _encode_image(image_path: Path) -> Tuple[str, str]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    mime, _ = mimetypes.guess_type(str(image_path))
    mime = mime or "image/png"
    with image_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return mime, b64


def _build_messages(prompt: str, image_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        mime, b64 = _encode_image(path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{b64}",
                    "detail": "high",
                },
            }
        )
    return [{"role": "user", "content": content}]


def call_vlm(
    prompt: str,
    image_paths: Sequence[str],
    cfg: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """
    调用多模态大模型。返回 (message_content, full_response_json)。
    """
    paths = [Path(p) for p in image_paths]
    messages = _build_messages(prompt, paths)
    base_url = str(cfg.get("base_url", "")).rstrip("/")
    api_key = cfg.get("api_key")
    if not base_url:
        raise LlmClientError("base_url is empty")
    if not api_key:
        raise LlmClientError("api_key is empty")

    payload: Dict[str, Any] = {
        "model": cfg.get("model", "gpt-4o"),
        "messages": messages,
        "temperature": cfg.get("temperature", 0.2),
        "max_tokens": cfg.get("max_tokens", 500),
    }
    if cfg.get("response_format", "json_object") == "json_object":
        payload["response_format"] = {"type": "json_object"}

    # Header/route 可配置，兼容非 OpenAI 网关
    header_name = cfg.get("api_key_header", "Authorization")
    header_prefix = cfg.get("api_key_prefix", "Bearer")
    header_value = f"{header_prefix} {api_key}" if header_prefix else api_key
    base_headers = {
        header_name: header_value,
        "Content-Type": "application/json",
    }
    extra_headers = cfg.get("extra_headers")
    if isinstance(extra_headers, dict):
        base_headers.update(extra_headers)

    url = base_url
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"

    def _send(headers: Dict[str, Any]) -> requests.Response:
        resp = requests.post(
            url,
            json=payload,
            timeout=cfg.get("timeout", 60),
            headers=headers,
        )
        resp.raise_for_status()
        return resp

    # 若 401 且提示未提供令牌，自动尝试常见头部写法
    tried_headers = [base_headers]
    fallback_headers = [
        {"Authorization": api_key, "Content-Type": "application/json"},
        {"X-API-Key": api_key, "Content-Type": "application/json"},
        {"api-key": api_key, "Content-Type": "application/json"},
        {"token": api_key, "Content-Type": "application/json"},
    ]
    if isinstance(extra_headers, dict) and extra_headers:
        for h in fallback_headers:
            h.update(extra_headers)

    resp = None
    try:
        resp = _send(base_headers)
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else ""
        status = exc.response.status_code if exc.response is not None else "error"
        if status == 401 and "未提供令牌" in body:
            for alt in fallback_headers:
                try:
                    resp = _send(alt)
                    break
                except requests.HTTPError:
                    continue
        if resp is None:
            raise LlmClientError(f"HTTP {status}: {body}") from exc

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise LlmClientError(f"Unexpected response schema: {data}") from exc
    return content, data


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    matches = re.finditer(r"\{.*\}", text, re.S)
    for m in matches:
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    raise LlmClientError("Failed to decode JSON from model output")


def decode_motion_weights(message: str) -> Dict[str, Any]:
    """
    将大模型返回的文本提取为字典，包含 body_part_weights。
    """
    parsed = _extract_json(message)
    if "body_part_weights" not in parsed:
        raise LlmClientError("body_part_weights missing in model output")
    weights = parsed["body_part_weights"] or {}
    return {
        "motion_type": parsed.get("motion_type", ""),
        "body_part_weights": weights,
        # 额外返回英文分析与理由（可选）
        "analysis_en": parsed.get("analysis_en", ""),
        "rationale_en": parsed.get("rationale_en", ""),
    }


def _clamp_weight(value: Any) -> int:
    try:
        num = float(value)
    except Exception:
        return 1
    num = max(1.0, min(5.0, num))
    return int(round(num))


def weights_to_csv_rows(
    weights: Dict[str, Any],
    frame_count: int,
) -> List[Dict[str, Any]]:
    frame_count = max(1, int(frame_count))
    body_weights = weights.get("body_part_weights", {}) if weights else {}
    row = {}
    for part, csv_name in BODY_PART_TO_CSV.items():
        row[csv_name] = _clamp_weight(body_weights.get(part, 1))
    rows: List[Dict[str, Any]] = []
    for idx in range(frame_count):
        rows.append({"frame": idx, **row})
    return rows


def write_csv(rows: List[Dict[str, Any]], output_path: os.PathLike) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(CSV_COLUMNS)]
    for r in rows:
        values = [str(r.get(col, "")) for col in CSV_COLUMNS]
        lines.append(",".join(values))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
