import argparse
import json
import cv2

from .pipeline import write_bundle


def main():
    p = argparse.ArgumentParser(
        description="Register unordered overlapping frames into an RGBA panorama"
    )
    p.add_argument("frames", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-pixels", type=int, default=10_000_000)
    p.add_argument(
        "--focal-px",
        type=float,
        help="use a cylindrical output at this reference-frame focal length in pixels",
    )
    args = p.parse_args()
    cv2.setNumThreads(1)
    try:
        r = write_bundle(
            args.frames, args.out, args.seed, args.max_pixels, args.focal_px
        )
    except (OSError, ValueError, cv2.error) as e:
        p.exit(2, f"lumenstitch: {e}\n")
    print(
        json.dumps(
            {k: r[k] for k in ("canvas", "root", "coverage_pixels", "optimization")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
