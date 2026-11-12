import math
import numpy as np


def project(H, points):
    xy = np.column_stack((points, np.ones(len(points)))) @ H.T
    w = xy[:, 2]
    result = np.full((len(points), 2), 1e9)
    good = np.abs(w) > 1e-10
    result[good] = xy[good, :2] / w[good, None]
    return result


def normalize(points):
    center = points.mean(axis=0)
    radius = np.linalg.norm(points - center, axis=1).mean()
    if radius < 1e-8:
        raise ValueError("coincident samples")
    s = math.sqrt(2) / radius
    T = np.array([[s, 0, -s * center[0]], [0, s, -s * center[1]], [0, 0, 1.0]])
    return (points - center) * s, T


def dlt(src, dst):
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("homography needs four paired points")
    x, A = normalize(np.asarray(src, dtype=float))
    y, B = normalize(np.asarray(dst, dtype=float))
    hom = np.column_stack((x, np.ones(len(x))))
    design = np.zeros((2 * len(x), 9))
    design[::2, :3] = hom
    design[1::2, 3:6] = hom
    design[::2, 6:] = -y[:, 0, None] * hom
    design[1::2, 6:] = -y[:, 1, None] * hom
    # A minimal 8x9 design needs the ninth right-singular vector (the null space).
    _, singular, vt = np.linalg.svd(design, full_matrices=len(src) == 4)
    if singular[7] < singular[0] * 1e-9:
        raise ValueError("degenerate homography samples")
    H = np.linalg.solve(B, vt[-1].reshape(3, 3) @ A)
    if abs(H[2, 2]) < 1e-12 or np.linalg.cond(H) > 1e12:
        raise ValueError("ill-conditioned homography")
    return H / H[2, 2]


def symmetric_error(H, src, dst):
    forward = np.linalg.norm(project(H, src) - dst, axis=1)
    backward = np.linalg.norm(project(np.linalg.inv(H), dst) - src, axis=1)
    return np.maximum(forward, backward)


def robust(src, dst, rng, threshold=3.5, max_trials=1100):
    if len(src) < 4:
        raise ValueError("insufficient correspondences")
    best = None
    score = (0, -math.inf)
    trials, limit = 0, max_trials
    while trials < limit:
        sample = rng.choice(len(src), 4, replace=False)
        trials += 1
        try:
            H = dlt(src[sample], dst[sample])
            error = symmetric_error(H, src, dst)
        except (np.linalg.LinAlgError, ValueError):
            continue
        mask = error <= threshold
        count = int(mask.sum())
        if count < 4:
            continue
        ranking = (count, -float(np.median(error[mask])))
        if ranking > score:
            score, best = ranking, mask
            probability = (count / len(src)) ** 4
            if probability >= 1:
                limit = trials
            elif probability > 0:
                limit = min(
                    limit,
                    max(trials, math.ceil(math.log(0.001) / math.log1p(-probability))),
                )
    if best is None:
        raise ValueError("no nondegenerate consensus")
    for _ in range(3):
        H = dlt(src[best], dst[best])
        error = symmetric_error(H, src, dst)
        keep = error <= threshold
        if np.array_equal(keep, best) or keep.sum() < 4:
            break
        best = keep
    H = dlt(src[best], dst[best])
    final = symmetric_error(H, src, dst) <= threshold
    return H, final, trials
