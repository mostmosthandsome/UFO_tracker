"""Convert named DR02 LaFAN CSVs to native UFO full and training PKLs.

Run from the repository root: python Data/convert_dr02_lafan.py --source DIR
Requires the repository's numpy, mujoco, joblib, loguru and omegaconf dependencies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ufo.motion.formats import RobotStateMotion, robot_state_to_ufo_motion, clip_ufo_motion_dict, validate_ufo_motion_dict
from ufo.tools.robot_config import load_robot_spec


def read_csv(path, spec):
    comments = {}
    with path.open(encoding="utf-8-sig") as stream:
        def rows():
            for line in stream:
                if line.lstrip().startswith("#"):
                    key, sep, value = line.lstrip()[1:].partition(":")
                    if sep:
                        comments[key.strip()] = value.strip()
                elif line.strip():
                    yield line
        reader = csv.reader(rows())
        header = next(reader)
        data = np.asarray(list(reader), dtype=np.float64)
    prefix = ["time", "root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw"]
    if len(header) != 31 or header[:8] != prefix or len(set(header)) != 31:
        raise ValueError(f"{path}: unexpected or duplicated columns")
    names = [name[4:] if name.startswith("dof_") else name for name in header[8:]]
    if set(names) != set(spec.control_joint_names) or len(set(names)) != 23:
        raise ValueError(f"{path}: joint names do not match DR02")
    if data.ndim != 2 or data.shape[1] != 31 or len(data) < 3 or not np.isfinite(data).all():
        raise ValueError(f"{path}: invalid numeric rows")
    times = data[:, 0]
    dt = np.diff(times)
    if (dt <= 0).any():
        raise ValueError(f"{path}: time must increase strictly")
    measured_fps = (len(times)-1) / (times[-1]-times[0])
    fps = float(comments.get("sample_rate", measured_fps))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{path}: invalid FPS")
    time_error = np.max(np.abs((times-times[0]) - np.arange(len(times))/fps))
    if time_error > 5e-6 or np.max(np.abs(dt-1/fps)) > 5e-6:
        raise ValueError(f"{path}: timestamps disagree with constant FPS={fps}; error={time_error}")
    quat = data[:, 4:8]
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if (norms < 1e-8).any() or np.max(np.abs(norms-1)) > 1e-3:
        raise ValueError(f"{path}: invalid root quaternion")
    motion = RobotStateMotion(
        motion_key=path.stem, root_pos=data[:, 1:4], root_quat=quat/norms,
        dof_pos=data[:, 8:], fps=fps, joint_names=names, source=str(path),
        metadata={"csv_comments": comments, "csv_joint_names": names,
                  "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    )
    record = robot_state_to_ufo_motion(motion, spec)
    violations = {}
    for index, name in enumerate(spec.control_joint_names):
        limits = spec.joint_ranges[name]
        if limits is not None:
            q = record["dof_pos"][:, index]
            excess = np.maximum(limits[0]-q, q-limits[1])
            count = int(np.count_nonzero(excess > 1e-4))
            if count:
                violations[name] = {"frames": count, "max_excess_rad": float(excess.max())}
    return record, dict(file=str(path), frames=len(data), fps=fps,
                       measured_fps=measured_fps, max_time_error_seconds=float(time_error),
                       input_joint_names=names, joint_limit_violations=violations,
                       sha256=motion.metadata["source_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    paths = sorted(args.source.expanduser().resolve().rglob("*.csv"))
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError("Need CSVs with unique motion names")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    filenames = ["dr02_lafan_full.pkl", "dr02_lafan_10s_clipped.pkl", "dr02_lafan.yaml", "conversion_report.json"]
    if any((output / name).exists() for name in filenames):
        raise FileExistsError("Output artifacts already exist; choose a new --output directory")
    spec = load_robot_spec(ROOT / "configs/robots/dr02_23dof.yaml")
    full, reports = {}, []
    for index, path in enumerate(paths):
        record, report = read_csv(path, spec)
        full[path.stem] = record
        reports.append(report)
        print(f"[{index+1}/{len(paths)}] {path.name}: {report['frames']} frames, {report['fps']:g} Hz", flush=True)
    validate_ufo_motion_dict(full, "dr02_lafan_full")
    train = clip_ufo_motion_dict(full, clip_seconds=10, stride_seconds=10,
                                keep_short=True, min_clip_seconds=1, source_name="dr02_lafan")
    full_frames = sum(len(x["dof_pos"]) for x in full.values())
    train_frames = sum(len(x["dof_pos"]) for x in train.values())
    for name, data in zip(filenames[:2], [full, train]):
        joblib.dump(data, output / name, compress=3)
    manifest = dict(robot_config=str(ROOT / "configs/robots/dr02_23dof.yaml"), datasets=[dict(
        name="dr02_lafan", format="ufo_pkl", weight=1.0,
        train_path="dr02_lafan_10s_clipped.pkl", inference_path="dr02_lafan_full.pkl")])
    OmegaConf.save(OmegaConf.create(manifest), output / filenames[2])
    summary = dict(source_directory=str(args.source.resolve()), full_motions=len(full),
                   full_frames=full_frames, training_clips=len(train), training_frames=train_frames,
                   omitted_short_tail_frames=full_frames-train_frames,
                   duration_hours=sum(len(x["dof_pos"])/x["fps"] for x in full.values())/3600,
                   fps_values=sorted({x["fps"] for x in full.values()}),
                   control_joint_names=list(spec.control_joint_names), body_names=list(spec.body_names),
                   clipping=dict(seconds=10, stride_seconds=10, keep_short=True, min_seconds=1),
                   files=reports)
    (output / filenames[3]).write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("files", "body_names", "control_joint_names")}, indent=2))


if __name__ == "__main__":
    main()
