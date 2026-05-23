"""FPS benchmark over the 6 ckpts on a fixed 200-frame video."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor,
)

REPO = Path(__file__).resolve().parents[2]


def bench_one(cfg: str, ckpt: str, video_dir: str, n_warm: int = 3, n_runs: int = 10) -> dict:
    predictor = build_efficienttam_video_predictor(
        cfg, ckpt, device="cuda",
        hydra_overrides_extra=["++model.compile_image_encoder=False"],
    )
    n_params = sum(p.numel() for p in predictor.parameters())
    inf_state = predictor.init_state(video_path=video_dir)

    frame_names = sorted(p for p in os.listdir(video_dir)
                         if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"])
    num_frames = len(frame_names)

    points = np.array([[210, 350]], dtype=np.float32)
    labels = np.array([1], np.int32)
    predictor.add_new_points_or_box(
        inference_state=inf_state, frame_idx=0, obj_id=1,
        points=points, labels=labels,
    )

    # warm
    with torch.inference_mode():
        for _ in range(n_warm):
            for _ in predictor.propagate_in_video(inf_state):
                pass

    # measure
    total = 0.0
    with torch.inference_mode():
        for _ in tqdm(range(n_runs), desc=f"bench {Path(ckpt).stem}"):
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in predictor.propagate_in_video(inf_state):
                pass
            torch.cuda.synchronize()
            total += time.time() - t0

    fps = n_runs * num_frames / total
    return {
        "n_params": n_params,
        "num_frames": num_frames,
        "n_runs": n_runs,
        "fps": fps,
        "ms_per_frame": 1000.0 / fps,
    }


def main():
    ckpts = [
        ("efficienttam_s",         "efficienttam_s.yaml"),
        ("efficienttam_s_2",       "efficienttam_s_2.yaml"),
        ("efficienttam_s_1",       "efficienttam_s_1.yaml"),
        ("efficienttam_ti",        "efficienttam_ti.yaml"),
        ("efficienttam_ti_2",      "efficienttam_ti_2.yaml"),
        ("efficienttam_s_512x512", "efficienttam_s_512x512.yaml"),
    ]

    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    video_dir = str(REPO / "notebooks/videos/bedroom")
    out = {}
    for name, cfg in ckpts:
        ckpt_path = str(REPO / f"checkpoints/{name}.pt")
        if not Path(ckpt_path).exists():
            print(f"[skip] {name}: ckpt not found")
            continue
        cfg_full = f"configs/efficienttam/{cfg}"
        print(f"\n=== {name} ===")
        out[name] = bench_one(cfg_full, ckpt_path, video_dir)
        print(f"  params={out[name]['n_params']/1e6:.1f}M  fps={out[name]['fps']:.1f}  ms/frame={out[name]['ms_per_frame']:.2f}")

    out_path = REPO / "outputs/ablation_generalization/fps_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] saved {out_path}")


if __name__ == "__main__":
    main()
