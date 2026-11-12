import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from lumenstitch.blend import blend, canvas
from lumenstitch.geometry import dlt, project, robust
from lumenstitch.pipeline import stitch, write_bundle

cv2.setNumThreads(1)
ROOT = Path(__file__).resolve().parents[1]


class RegistrationTests(unittest.TestCase):
    def test_dlt_and_collinear_rejection(self):
        rng = np.random.default_rng(9)
        src = rng.uniform(100, 300, (15, 2))
        H = np.array([[1.03, 0.03, 11], [-0.01, 0.96, -4], [0.00018, -0.00012, 1]])
        dst = project(H, src)
        for count in (4, 15):
            recovered = dlt(src[:count], dst[:count])
            np.testing.assert_allclose(project(recovered, src), dst, atol=1e-7)
        x = np.array([[i, i] for i in range(7)])
        with self.assertRaises(ValueError):
            dlt(x, x)

    def test_ransac_with_planted_outliers(self):
        rng = np.random.default_rng(45)
        truth = np.array([[1, 0.025, 50], [-0.018, 1.02, 9], [0.00012, 0.00007, 1]])
        src = rng.uniform(0, 650, (150, 2))
        dst = project(truth, src) + rng.normal(0, 0.3, (150, 2))
        dst[:68] = rng.uniform(0, 650, (68, 2))
        estimate, mask, _ = robust(src, dst, rng, threshold=2)
        verify = rng.uniform(0, 650, (70, 2))
        self.assertLess(
            np.median(
                np.linalg.norm(
                    project(estimate, verify) - project(truth, verify), axis=1
                )
            ),
            0.25,
        )
        self.assertGreaterEqual(mask[68:].sum(), 78)
        self.assertLessEqual(mask[:68].sum(), 1)

    def test_held_out_geometry_on_unordered_images(self):
        rng = np.random.default_rng(88)
        scene = cv2.GaussianBlur(
            rng.integers(10, 245, (350, 650, 3), dtype=np.uint8), (3, 3), 0
        )
        for i in range(100):
            pt = tuple(map(int, rng.integers([10, 10], [630, 320])))
            cv2.circle(
                scene,
                pt,
                int(rng.integers(2, 12)),
                tuple(map(int, rng.integers(20, 240, 3))),
                -1,
            )
        poses = [
            np.array([[1, 0.01, 250], [0, 1, 32], [0.00006, 0, 1]]),
            np.array([[1, 0, 5], [0, 1, 25], [0, 0, 1]]),
            np.array([[0.99, -0.02, 135], [0.01, 1, 28], [-0.00006, 0, 1]]),
        ]
        frames = [
            cv2.warpPerspective(scene, np.linalg.inv(h), (320, 255)) for h in poses
        ]
        frames[2] = np.round(frames[2].astype(float) * 0.8).astype(np.uint8)
        out, r = stitch(frames)
        sample = np.array([[50, 40], [280, 40], [50, 210], [280, 210], [160, 120]])
        for i, H in enumerate(r["image_to_reference"]):
            expected = np.linalg.inv(poses[r["root"]]) @ poses[i]
            errors = np.linalg.norm(
                project(np.asarray(H), sample) - project(expected, sample), axis=1
            )
            self.assertLess(errors.max(), 1.3)
        self.assertGreater(np.count_nonzero(out[:, :, 3]), 130000)
        self.assertTrue(np.isfinite(r["gains"]).all())

    def test_black_pixels_and_exposure(self):
        base = np.full((110, 140, 3), 120, dtype=np.uint8)
        base[20:45, 20:45] = 0
        other = np.round(base * 0.75).astype(np.uint8)
        result, projection, gains = blend([base, other], [np.eye(3), np.eye(3)], 0)
        self.assertEqual(projection["kind"], "planar")
        self.assertEqual(result[32, 32, 3], 255)
        self.assertLessEqual(int(result[32, 32, 0]), 1)
        self.assertAlmostEqual(gains[1], 4 / 3, delta=0.015)
        self.assertAlmostEqual(float(result[60, 60, 0]), 120, delta=1)
        cylinder, meta, _ = blend([base, other], [np.eye(3), np.eye(3)], 0, focal=150)
        self.assertEqual(meta["kind"], "cylindrical")
        self.assertGreater(np.count_nonzero(cylinder[:, :, 3]), 12000)

    def test_rejects_horizon_crossings_and_oversized_output(self):
        im = np.zeros((60, 80, 3), np.uint8)
        center = np.array([40, 30])
        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            canvas([im], [np.eye(3)], 100, center)
        H = np.eye(3)
        H[2] = [0.1, 0, -1]
        with self.assertRaisesRegex(ValueError, "horizon"):
            canvas([im], [H], 9000, center)
        with self.assertRaisesRegex(ValueError, "focal"):
            canvas([im], [np.eye(3)], 9000, center, -2)

    def test_disconnected_frame_does_not_publish_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            cv2.imwrite(str(p / "blank.png"), np.zeros((80, 80, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, "disconnected"):
                write_bundle([p / "blank.png"] * 2, p / "out", 4, 100000)
            self.assertFalse((p / "out").exists())

    def test_real_mission_frames_through_cli(self):
        folder = ROOT / "examples/apollo"
        sources = json.loads((folder / "sources.json").read_text())
        frames = [folder / (s["frame"] + ".jpg") for s in sources]
        for f, s in zip(frames, sources):
            self.assertEqual(
                hashlib.sha256(f.read_bytes()).hexdigest(), s["sample_sha256"]
            )
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "result"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lumenstitch",
                    *frames,
                    "--focal-px",
                    "810",
                    "--out",
                    dest,
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
            )
            report = json.loads((dest / "alignment.json").read_text())
            png = cv2.imread(str(dest / "panorama.png"), cv2.IMREAD_UNCHANGED)
            self.assertEqual(png.shape[2], 4)
            self.assertGreater(
                sum(p["inliers"] for p in report["pairs"] if p["accepted"]), 350
            )
            self.assertGreater(report["coverage_pixels"], 600000)
            self.assertLess(report["optimization"]["rms_component_after_px"], 2.0)
