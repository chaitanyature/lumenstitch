from dataclasses import dataclass
from itertools import combinations
from collections import deque

import cv2
import numpy as np
from scipy.optimize import least_squares

from .geometry import project, robust, symmetric_error


@dataclass
class Edge:
    i: int
    j: int
    H: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    median: float


def correspondences(images, seed=17):
    sift = cv2.SIFT_create(nfeatures=2400, contrastThreshold=0.025)
    features = []
    for im in images:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        keys, desc = sift.detectAndCompute(gray, None)
        features.append(
            (np.array([k.pt for k in keys], dtype=float).reshape(-1, 2), desc)
        )
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def ratio(left, right):
        return {
            pair[0].queryIdx: pair[0].trainIdx
            for pair in matcher.knnMatch(left, right, k=2)
            if len(pair) == 2 and pair[0].distance < 0.76 * pair[1].distance
        }

    edges, diagnostics = [], []
    for i, j in combinations(range(len(images)), 2):
        pi, di = features[i]
        pj, dj = features[j]
        report = {"pair": [i, j], "matches": 0, "inliers": 0, "accepted": False}
        diagnostics.append(report)
        if di is None or dj is None or len(di) < 2 or len(dj) < 2:
            continue
        left, right = ratio(di, dj), ratio(dj, di)
        matches = [(a, b) for a, b in left.items() if right.get(b) == a]
        report["matches"] = len(matches)
        if len(matches) < 12:
            continue
        src = pi[[a for a, b in matches]]
        dst = pj[[b for a, b in matches]]
        try:
            H, keep, trials = robust(src, dst, np.random.default_rng([seed, i, j]))
        except (ValueError, np.linalg.LinAlgError):
            continue
        count = int(keep.sum())
        report.update(inliers=count, trials=trials)
        if count < 12 or count / len(matches) < 0.25:
            continue
        a, b = src[keep], dst[keep]
        # Reject accidental consensus confined to a tiny marking or narrow line.
        fractions = [
            cv2.contourArea(cv2.convexHull(p.astype(np.float32)))
            / (im.shape[0] * im.shape[1])
            for p, im in [(a, images[i]), (b, images[j])]
        ]
        if min(fractions) < 0.008:
            continue
        median = float(np.median(symmetric_error(H, a, b)))
        report.update(accepted=True, median_symmetric_px=median)
        edges.append(Edge(i, j, H, a, b, median))
    return edges, diagnostics


def initialize(count, edges):
    neighbors = [set() for _ in range(count)]
    for e in edges:
        neighbors[e.i].add(e.j)
        neighbors[e.j].add(e.i)

    def distances(i):
        dist = {i: 0}
        q = deque([i])
        while q:
            at = q.popleft()
            for nxt in sorted(neighbors[at]):
                if nxt not in dist:
                    dist[nxt] = dist[at] + 1
                    q.append(nxt)
        return dist

    dist = [distances(i) for i in range(count)]
    if len(dist[0]) != count:
        raise ValueError(
            f"disconnected match graph; frames unreachable from 0: {sorted(set(range(count)) - set(dist[0]))}"
        )
    root = min(
        range(count), key=lambda i: (max(dist[i].values()), sum(dist[i].values()), i)
    )
    poses = {root: np.eye(3)}
    tree = set()
    while len(poses) < count:
        e = max(
            (e for e in edges if (e.i in poses) != (e.j in poses)),
            key=lambda e: len(e.src) / (1 + e.median),
        )
        if e.i in poses:
            poses[e.j] = poses[e.i] @ np.linalg.inv(e.H)
        else:
            poses[e.i] = poses[e.j] @ e.H
        tree.add((e.i, e.j))
    return root, [poses[i] / poses[i][2, 2] for i in range(count)], tree


def refine(images, poses, root, edges, tree):
    # Optimize in image-centered normalized coordinates, measuring residuals in pixels.
    frames = []
    scales = []
    for img in images:
        h, w = img.shape[:2]
        s = max(h, w) / 2
        frames.append(
            np.array([[1 / s, 0, -w / (2 * s)], [0, 1 / s, -h / (2 * s)], [0, 0, 1]])
        )
        scales.append(s)
    G = [frames[root] @ H @ np.linalg.inv(T) for H, T in zip(poses, frames)]
    G = [g / g[2, 2] for g in G]
    free = [i for i in range(len(images)) if i != root]
    measurements = []
    ignored = []
    for e in edges:
        predicted = np.linalg.inv(poses[e.j]) @ poses[e.i]
        mismatch = float(np.median(symmetric_error(predicted, e.src, e.dst)))
        if (e.i, e.j) not in tree and mismatch > 12:
            ignored.append([e.i, e.j])
            continue
        # Even sampling caps optimization size without biasing toward descriptor order.
        take = np.linspace(0, len(e.src) - 1, min(140, len(e.src)), dtype=int)
        measurements.append(
            (
                e.i,
                e.j,
                project(frames[e.i], e.src[take]),
                project(frames[e.j], e.dst[take]),
            )
        )

    def unpack(x):
        mats = G.copy()
        for j, i in enumerate(free):
            mats[i] = np.append(x[j * 8 : (j + 1) * 8], 1).reshape(3, 3)
        return mats

    def residuals(x):
        gs = unpack(x)
        parts = []
        for i, j, a, b in measurements:
            try:
                rel = np.linalg.solve(gs[j], gs[i])
                inv = np.linalg.inv(rel)
                parts.extend(
                    [
                        (project(rel, a) - b).ravel() * scales[j],
                        (project(inv, b) - a).ravel() * scales[i],
                    ]
                )
            except np.linalg.LinAlgError:
                parts.append(np.full(len(a) * 4, 1e6))
        return np.concatenate(parts)

    x0 = np.concatenate([G[i].ravel()[:8] for i in free])
    res = least_squares(
        residuals, x0, loss="soft_l1", f_scale=2, max_nfev=70, x_scale="jac"
    )
    out = [np.linalg.inv(frames[root]) @ g @ T for g, T in zip(unpack(res.x), frames)]

    def rms(x):
        return float(np.sqrt(np.mean(residuals(x) ** 2)))

    return [h / h[2, 2] for h in out], {
        "rms_component_before_px": rms(x0),
        "rms_component_after_px": rms(res.x),
        "rejected_loop_edges": ignored,
        "evaluations": res.nfev,
    }
