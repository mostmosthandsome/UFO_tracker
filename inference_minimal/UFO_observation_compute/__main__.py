"""Convert named RobotState NPZ into encode-ready observation tensors."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .reference import ReferenceObservationComputer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True, help="Explicit observation/robot settings JSON; see README")
    parser.add_argument("--motion", type=Path, required=True, help="RobotState NPZ")
    parser.add_argument("--fps", type=float, help="Override source fps")
    parser.add_argument("--control-fps", type=float, help="Interpolate at control times, excluding the source end time like UFO tracking")
    parser.add_argument("--drop-first", action="store_true", help="Apply UFO tracking's next-frame alignment")
    parser.add_argument("--output", type=Path, required=True, help="New .pt file containing the observation dictionary")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    metadata = json.loads(args.metadata.read_text())
    computer = ReferenceObservationComputer(args.xml, **metadata)
    with np.load(args.motion, allow_pickle=False) as motion:
        fps = args.fps if args.fps is not None else float(np.asarray(motion["fps"]).item())
        if not math.isfinite(fps) or fps <= 0:
            parser.error("Source FPS must be finite and positive")
        times = None
        if args.control_fps is not None:
            if not math.isfinite(args.control_fps) or args.control_fps <= 0:
                parser.error("--control-fps must be finite and positive")
            duration = (len(motion["dof_pos"]) - 1) / fps
            times = torch.arange(math.ceil(duration * args.control_fps)) / args.control_fps
            if not times.numel():
                parser.error("Control-time sampling requires a sequence with positive duration")
        optional = {key: motion[key] for key in ("dof_vel", "body_vel", "body_ang_vel") if key in motion}
        result = computer.from_robot_state(
            motion["dof_pos"], motion["root_pos"], motion["root_quat"], fps,
            joint_names=motion["joint_names"].tolist() if "joint_names" in motion else None,
            sample_times=times, drop_first=args.drop_first, **optional,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.cpu() for key, value in result.items()}, args.output)
    print("Body order:", computer.body_names)
    print("Output:", args.output)
    print("Shapes:", {key: tuple(value.shape) for key, value in result.items()})


if __name__ == "__main__":
    main()
