"""`viobench simulate` with faster autopilot limits, for server-like test flights.

The stock generator flies 14-22 m per 60 s (cruise 0.45-0.8 m/s, acceleration
0.7 m/s^2, yaw rate 0.8 rad/s). The course server's flights cover 40-80 m per
60 s. This wrapper adds overrides to the task spec of viobench's world template
and then runs the normal `viobench simulate` command with the remaining arguments.

    python scripts/simulate_fast.py --cruise 1.5 2.5 --accel 2.0 --jerk 6 --yaw-rate 1.6 -- \
        -o data/val-fast --scene-names ProcTHOR-Train-30 --episodes 2 --seed 4000 --hover 0 --no-video
"""
import argparse
from pathlib import Path
import sys
import tempfile

from viobench.tools import cli
from viobench.tools.datasets.generate import simulation

MARKER = "static constexpr TI INITIALIZATION_HOLD_STEPS = @HOLD_STEPS@;"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cruise", nargs=2, type=float, default=[1.5, 2.5], metavar=("MIN", "MAX"))
    parser.add_argument("--accel", type=float, default=2.0)
    parser.add_argument("--jerk", type=float, default=6.0)
    parser.add_argument("--yaw-rate", type=float, default=1.6)
    parser.add_argument("simulate", nargs=argparse.REMAINDER, help="arguments after -- go to viobench simulate")
    args = parser.parse_args()
    rest = args.simulate[1:] if args.simulate[:1] == ["--"] else args.simulate
    text = simulation.TEMPLATE.read_text()
    if text.count(MARKER) != 1:
        raise SystemExit("unexpected world template; refusing to patch")
    overrides = (f"static constexpr T CRUISE_SPEED_MIN = {args.cruise[0]!r};\n"
                 f"        static constexpr T CRUISE_SPEED_MAX = {args.cruise[1]!r};\n"
                 f"        static constexpr T MAX_ACCELERATION = {args.accel!r};\n"
                 f"        static constexpr T MAX_JERK = {args.jerk!r};\n"
                 f"        static constexpr T MAX_YAW_RATE = {args.yaw_rate!r};\n        ")
    with tempfile.TemporaryDirectory() as scratch:
        template = Path(scratch) / "hyperdrone_world_fast.h.in"
        template.write_text(text.replace(MARKER, overrides + MARKER))
        simulation.TEMPLATE = template
        print(f"fast autopilot: cruise {args.cruise}, accel {args.accel}, jerk {args.jerk}, yaw rate {args.yaw_rate}", flush=True)
        sys.argv = ["viobench", "simulate", *rest]
        return cli.main()


if __name__ == "__main__":
    sys.exit(main())
