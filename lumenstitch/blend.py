import cv2
import numpy as np

from .geometry import project


def canvas(images, poses, limit, center, focal=None):
    if focal is not None and (not np.isfinite(focal) or focal <= 0):
        raise ValueError("cylindrical focal length must be positive")
    points = []
    for img, H in zip(images, poses):
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
        depth = np.column_stack((corners, np.ones(4))) @ H[2, :]
        if np.min(depth) <= 0 <= np.max(depth):
            raise ValueError(
                "projective horizon intersects a frame; use a shorter sweep"
            )
        if focal is None:
            points.extend(project(H, corners))
        else:
            border = np.vstack(
                [
                    np.linspace(a, b, 257)
                    for a, b in zip(corners, np.roll(corners, -1, axis=0))
                ]
            )
            x, y = (project(H, border) - center).T
            angle = np.arctan2(x, focal)
            points.extend(np.column_stack((focal * angle, y * np.cos(angle))))
    coords = np.array(points)
    if not np.isfinite(coords).all() or abs(coords).max() > 1e7:
        raise ValueError("invalid projected coordinates")
    low = np.floor(coords.min(axis=0)).astype(int)
    high = np.ceil(coords.max(axis=0)).astype(int)
    width, height = map(int, high - low + 1)
    if width * height > limit or max(width, height) >= 32767:
        raise ValueError(f"panorama canvas {width}x{height} exceeds limit")
    return (width, height), low


def warp(img, H, size, low, center, focal):
    source_mask = np.ones(img.shape[:2], np.uint8)
    if focal is None:
        shift = np.array([[1, 0, -low[0]], [0, 1, -low[1]], [0, 0, 1.0]])
        transform = shift @ H
        return (
            cv2.warpPerspective(img, transform, size, borderMode=cv2.BORDER_REPLICATE),
            cv2.warpPerspective(source_mask, transform, size, flags=cv2.INTER_NEAREST),
        )
    width, height = size
    angle = (np.arange(width) + low[0]) / focal
    x = focal * np.tan(angle) + center[0]
    inverse = np.linalg.inv(H)
    output = np.empty((height, width, 3), np.uint8)
    mask = np.empty((height, width), np.uint8)
    for start in range(0, height, 96):
        end = min(start + 96, height)
        y = (np.arange(start, end)[:, None] + low[1]) / np.cos(angle) + center[1]
        denom = inverse[2, 0] * x + inverse[2, 1] * y + inverse[2, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            mx = (
                (inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]) / denom
            ).astype(np.float32)
            my = (
                (inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]) / denom
            ).astype(np.float32)
        output[start:end] = cv2.remap(
            img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        mask[start:end] = cv2.remap(source_mask, mx, my, cv2.INTER_NEAREST)
    return output, mask


def exposure(images, masks, root):
    rows = []
    values = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            vi = cv2.cvtColor(images[i][::4, ::4], cv2.COLOR_BGR2GRAY).astype(float)
            vj = cv2.cvtColor(images[j][::4, ::4], cv2.COLOR_BGR2GRAY).astype(float)
            keep = (
                (masks[i][::4, ::4] > 0)
                & (masks[j][::4, ::4] > 0)
                & (vi > 12)
                & (vj > 12)
                & (vi < 242)
                & (vj < 242)
            )
            if keep.sum() < 60:
                continue
            row = np.zeros(len(images))
            row[i] = 1
            row[j] = -1
            weight = min(1, float(keep.sum()) / 700) ** 0.5
            rows.append(row * weight)
            values.append(np.median(np.log(vj[keep] / vi[keep])) * weight)
    # Weak priors regularize isolated exposure estimates; fix the reference gain to one.
    free = [i for i in range(len(images)) if i != root]
    A = np.vstack([*rows, *np.eye(len(images)) * 0.01])[:, free]
    vec = np.array([*values, *np.zeros(len(images))])
    logs = np.zeros(len(images))
    logs[free] = np.linalg.lstsq(A, vec, rcond=None)[0]
    return np.exp(np.clip(logs, np.log(0.5), np.log(2.0)))


def pyramid(x, depth):
    out = [x]
    for i in range(depth):
        out.append(cv2.pyrDown(out[-1]))
    return out


def up(x, shape):
    return cv2.pyrUp(x, dstsize=(shape[1], shape[0]))


def blend(images, poses, root, limit=10_000_000, levels=5, focal=None):
    h, w = images[root].shape[:2]
    center = np.array([(w - 1) / 2, (h - 1) / 2])
    size, low = canvas(images, poses, limit, center, focal)
    masks = []
    warps = []
    for img, H in zip(images, poses):
        image, mask = warp(img, H, size, low, center, focal)
        masks.append(mask)
        warps.append(image)
    gains = exposure(warps, masks, root)
    depth = max(0, min(levels, int(np.log2(min(size))) - 2))
    accum = []
    weights = []
    alpha = np.zeros(masks[0].shape, dtype=np.uint8)
    for img, mask, gain in zip(warps, masks, gains):
        alpha |= mask
        padded = np.pad(mask, 1)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        gp = pyramid(img.astype(np.float32) * np.float32(gain), depth)
        wp = pyramid(distance, depth)
        for i, coarse in enumerate(gp):
            lap = coarse if i == depth else coarse - up(gp[i + 1], coarse.shape)
            if len(accum) <= i:
                accum.append(np.zeros_like(lap))
                weights.append(np.zeros_like(wp[i]))
            accum[i] += lap * wp[i][:, :, None]
            weights[i] += wp[i]
    bands = [a / np.maximum(w[:, :, None], 1e-7) for a, w in zip(accum, weights)]
    result = bands[-1]
    for fine in reversed(bands[:-1]):
        result = up(result, fine.shape) + fine
    rgb = np.clip(np.rint(result), 0, 255).astype(np.uint8)
    rgb[alpha == 0] = 0
    projection = {
        "kind": "planar" if focal is None else "cylindrical",
        "focal_px": focal,
        "principal_point": center.tolist(),
        "canvas_origin": low.tolist(),
    }
    return np.dstack([rgb, alpha * 255]), projection, gains
