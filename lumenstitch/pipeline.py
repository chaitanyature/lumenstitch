import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .alignment import correspondences, initialize, refine
from .blend import blend


def stitch(images, seed=17, max_pixels=10_000_000, focal=None):
    if not 2 <= len(images) <= 10:
        raise ValueError("supply 2..10 frames")
    for im in images:
        if (
            im is None
            or im.dtype != np.uint8
            or im.ndim != 3
            or im.shape[2] != 3
            or min(im.shape[:2]) < 32
        ):
            raise ValueError(
                "expected 8-bit BGR images at least 32 pixels wide and tall"
            )
    edges, pairs = correspondences(images, seed)
    root, initial, tree = initialize(len(images), edges)
    poses, optimization = refine(images, initial, root, edges, tree)
    output, projection, gains = blend(images, poses, root, max_pixels, focal=focal)
    report = {
        "format": 1,
        "root": root,
        "pairs": pairs,
        "tree": [list(edge) for edge in sorted(tree)],
        "optimization": optimization,
        "gains": gains.tolist(),
        "canvas": [output.shape[1], output.shape[0]],
        "coverage_pixels": int(np.count_nonzero(output[:, :, 3])),
        "image_to_reference": [h.tolist() for h in poses],
        "projection": projection,
    }
    return output, report


def write_bundle(paths, out, seed, max_pixels, focal=None):
    out = Path(out)
    if out.exists():
        raise ValueError(f"output must be a new directory: {out}")
    frames = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]
    image, report = stitch(frames, seed, max_pixels, focal)
    report["frames"] = [Path(p).name for p in paths]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".lumen-") as temp:
        bundle = Path(temp) / "bundle"
        bundle.mkdir()
        if not cv2.imwrite(str(bundle / "panorama.png"), image):
            raise OSError("PNG encoder failed")
        (bundle / "alignment.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        os.rename(bundle, out)
    return report
