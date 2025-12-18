import argparse
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Extract evenly spaced frame sequences from videos.")
    parser.add_argument(
        "--config",
        type=str,
        default="vlm/cfg/frame_extract.yaml",
        help="Path to frame extraction YAML config.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override input directory or single video path (takes precedence over config).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (takes precedence over config).",
    )
    return parser.parse_args()


def load_config(cfg_path: str) -> dict:
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("input_dir", "videos")
    cfg.setdefault("output_dir", "frames")
    cfg.setdefault("num_frames", 32)
    cfg.setdefault("output_format", "png")
    cfg.setdefault("video_exts", [".mp4", ".mov", ".avi", ".mkv"])
    cfg.setdefault("resize", {"enabled": True, "width": 256, "height": 256})
    cfg.setdefault("combine", {"enabled": False, "output_name": "", "keep_frames": True})

    return cfg


def list_videos(input_dir: Path, exts: List[str]) -> List[Path]:
    exts = [e.lower() for e in exts]
    if input_dir.is_file():
        return [input_dir] if input_dir.suffix.lower() in exts else []

    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径既不是文件也不是目录: {input_dir}")

    return [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in exts and p.is_file()]


def evenly_spaced_indices(frame_count: int, num_frames: int) -> List[int]:
    if frame_count <= 0:
        return []
    num = max(1, min(num_frames, frame_count))
    # 线性均匀采样帧索引（含首尾）
    idx = np.linspace(0, frame_count - 1, num=num, dtype=int)
    return sorted(set(idx.tolist()))


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def extract_frames(video_path: Path, out_dir: Path, num_frames: int, resize_cfg: dict, output_format: str):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 无法打开视频: {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = evenly_spaced_indices(frame_count, num_frames)
    if len(target_indices) == 0:
        print(f"[WARN] 视频无有效帧: {video_path}")
        cap.release()
        return

    resize_enabled = bool(resize_cfg.get("enabled", True))
    resize_w = int(resize_cfg.get("width", 256))
    resize_h = int(resize_cfg.get("height", 256))

    ensure_dir(out_dir)
    target_ptr = 0
    current_idx = 0

    while target_ptr < len(target_indices):
        ret, frame = cap.read()
        if not ret:
            break

        if current_idx == target_indices[target_ptr]:
            if resize_enabled and resize_w > 0 and resize_h > 0:
                frame = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

            frame_name = f"{video_path.stem}_{target_ptr:04d}.{output_format}"
            out_path = out_dir / frame_name
            cv2.imwrite(str(out_path), frame)
            target_ptr += 1

        current_idx += 1

    cap.release()


def stitch_frames(frames_dir: Path, output_path: Path, output_format: str):
    """
    将 frames_dir 下的帧按文件名排序后横向拼接输出。
    假设所有帧已被统一尺寸；若尺寸不一致则跳过该帧。
    """
    frame_files = sorted([p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in [f".{output_format.lower()}"]])
    if len(frame_files) == 0:
        print(f"[WARN] 拼接失败，未找到帧文件: {frames_dir}")
        return

    imgs = []
    base_hw = None
    for f in frame_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w, _ = img.shape
        if base_hw is None:
            base_hw = (h, w)
        if (h, w) != base_hw:
            print(f"[WARN] 跳过尺寸不一致的帧: {f.name}")
            continue
        imgs.append(img)

    if len(imgs) == 0:
        print(f"[WARN] 没有可拼接的有效帧: {frames_dir}")
        return

    combined = cv2.hconcat(imgs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), combined)
    if not success:
        raise RuntimeError(f"组合帧保存失败，无法写入文件（检查扩展名/路径）: {output_path}")
    print(f"[INFO] 拼接完成: {output_path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    input_dir = Path(args.input_dir) if args.input_dir else Path(cfg["input_dir"])
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"])
    combine_cfg = cfg.get("combine", {"enabled": False})

    if not input_dir.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_dir}")

    videos = list_videos(input_dir, cfg["video_exts"])
    if len(videos) == 0:
        print(f"[WARN] 未找到可处理的视频文件: {input_dir}")
        return

    print(f"[INFO] 共 {len(videos)} 个视频，将为每个视频提取 {cfg['num_frames']} 帧")

    for vid in videos:
        per_video_out = output_dir / vid.stem
        print(f"[INFO] 处理: {vid.name} -> {per_video_out}")
        extract_frames(
            video_path=vid,
            out_dir=per_video_out,
            num_frames=int(cfg["num_frames"]),
            resize_cfg=cfg.get("resize", {}),
            output_format=str(cfg.get("output_format", "png")),
        )

        if combine_cfg.get("enabled", False):
            output_fmt = str(cfg.get("output_format", "png"))
            custom_name = str(combine_cfg.get("output_name", "")).strip()
            combined_name = custom_name if custom_name else f"{vid.stem}_combined.{output_fmt}"
            combined_path = output_dir / combined_name
            stitch_frames(
                frames_dir=per_video_out,
                output_path=combined_path,
                output_format=output_fmt,
            )
            if not combine_cfg.get("keep_frames", True):
                # 删除中间帧，仅保留组合帧
                for f in per_video_out.iterdir():
                    if f.is_file():
                        try:
                            f.unlink()
                        except OSError:
                            pass
                try:
                    per_video_out.rmdir()
                except OSError:
                    pass

    print("[INFO] 提取完成")


if __name__ == "__main__":
    main()
