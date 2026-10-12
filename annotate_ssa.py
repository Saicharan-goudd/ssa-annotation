#!/usr/bin/env python3
"""
SSA star/streak annotation pipeline (FITS -> 1024x1024 tiles -> YOLO-seg labels).

Steps
  1. Read 16-bit FITS, estimate + subtract smooth background, robust noise (MAD)
  2. Detect on a lightly smoothed copy with hysteresis thresholding
  3. Classify each connected component: star (class 0) / streak (class 1)
  4. Pad to a multiple of 1024 (no cropping -> no data loss), tile, write
     YOLO-Ultralytics segmentation labels (polygons clipped per tile)
  5. Repatch tiles to full size, verify lossless round-trip, export overlay + catalog

Usage:
  python annotate_ssa.py --input Datasets_Assessment --out output
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np
from astropy.io import fits
from scipy import ndimage as ndi

TILE = 1024
CLASS_NAMES = ["star", "streak"]

# ---- tunable parameters (document any change you make after inspecting real data)
P = dict(
    bkg_block=128,        # px, block size for background median grid
    smooth_sigma=1.0,     # px, Gaussian for detection image only
    seed_sigma=5.0,       # peak (smoothed) must exceed this many noise sigmas
    grow_sigma=2.5,       # region grown down to this threshold
    min_area=4,           # px, reject smaller (hot pixels / noise)
    close_kernel=5,       # px, closing to rejoin broken/faint streaks
    streak_elong=3.0,     # major/minor axis ratio -> streak
    streak_min_len=12.0,  # px, major axis length for the elongation rule
    fat_elong=2.0,        # relaxed ratio for short/fat streaks ...
    fat_min_len=10.0,     # ... which must also be this long
    fat_min_fill=0.6,     # ... and fill >= this fraction of their min-area rectangle
    long_len=40.0,        # anything this long and >=2:1 is a streak
    ambiguous=(1.6, 2.4), # elongation band flagged for manual review
)


# ------------------------------------------------------------------ I/O
def read_fits(path):
    with fits.open(path) as hdul:
        for h in hdul:
            if h.data is not None and h.data.ndim == 2:
                return np.asarray(h.data).astype(np.float32)
    raise ValueError(f"No 2-D image HDU in {path}")


# ------------------------------------------------------------------ pre-processing
def estimate_background(img, block):
    H, W = img.shape
    gy, gx = int(np.ceil(H / block)), int(np.ceil(W / block))
    grid = np.empty((gy, gx), np.float32)
    for j in range(gy):
        for i in range(gx):
            grid[j, i] = np.median(img[j * block:(j + 1) * block, i * block:(i + 1) * block])
    grid = ndi.median_filter(grid, size=3, mode="nearest")
    return cv2.resize(grid, (W, H), interpolation=cv2.INTER_CUBIC)


def mad_sigma(a):
    return 1.4826 * np.median(np.abs(a - np.median(a)))


def make_stretch(res, sigma):
    """asinh stretch fixed per image so every tile shares one mapping."""
    rmax = max(float(np.percentile(res, 99.99)), 10 * sigma)
    norm = np.arcsinh(rmax / (2 * sigma))

    def f(x):
        y = np.arcsinh(np.clip(x, -3 * sigma, None) / (2 * sigma)) / norm
        return np.clip(y * 255, 0, 255).astype(np.uint8)

    return f


# ------------------------------------------------------------------ detection + classification
def detect(res, sigma_pix):
    sm = cv2.GaussianBlur(res, (0, 0), P["smooth_sigma"])
    s_sm = mad_sigma(sm)
    low = sm > P["grow_sigma"] * s_sm
    k = P["close_kernel"]
    low = cv2.morphologyEx(low.astype(np.uint8), cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    lab, n = ndi.label(low, structure=np.ones((3, 3)))   # 8-connectivity
    if n == 0:
        return lab, 0, s_sm
    peaks = ndi.maximum(sm, lab, index=np.arange(1, n + 1))
    areas = ndi.sum(low, lab, index=np.arange(1, n + 1))
    keep = (peaks >= P["seed_sigma"] * s_sm) & (areas >= P["min_area"])
    remap = np.zeros(n + 1, np.int32)
    remap[1:][keep] = np.arange(1, keep.sum() + 1)
    return remap[lab], int(keep.sum()), s_sm


def shape_features(mask):
    ys, xs = np.nonzero(mask)
    area = len(xs)
    if area < 3:
        return dict(area=area, major=float(area), minor=1.0, elong=1.0, fill=1.0)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1]
    major, minor = 4 * np.sqrt(max(ev[0], 1e-6)), max(4 * np.sqrt(max(ev[1], 1e-6)), 1.0)
    (w, h) = cv2.minAreaRect(pts)[1]
    fill = area / max(w * h, 1.0)
    return dict(area=area, major=float(major), minor=float(minor),
                elong=float(major / minor), fill=float(fill))


def classify(f):
    e, L = f["elong"], f["major"]
    if e >= P["streak_elong"] and L >= P["streak_min_len"]:
        return 1
    if L >= P["long_len"] and e >= 2.0:
        return 1
    if e >= P["fat_elong"] and L >= P["fat_min_len"] and f["fill"] >= P["fat_min_fill"]:
        return 1
    return 0


# ------------------------------------------------------------------ YOLO polygons
def polygons_for_tile(sub):
    """sub: bool mask clipped to one tile -> list of polygons (Nx2 px coords)."""
    pad = np.pad(sub.astype(np.uint8), 1)
    cnts, _ = cv2.findContours(pad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in cnts:
        c = cv2.approxPolyDP(c, 0.7, True).reshape(-1, 2) - 1
        if len(c) >= 3 and cv2.contourArea(c.astype(np.float32)) >= 1:
            polys.append(c.astype(np.float32))
        else:  # tiny object: use its pixel bounding box so it still has area
            ys, xs = np.nonzero(sub)
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            polys.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32))
            break
    return polys


# ------------------------------------------------------------------ main per-image routine
def process(path, out, save16=False):
    stem = os.path.splitext(os.path.basename(path))[0]
    raw = read_fits(path)
    H, W = raw.shape

    # 1. pre-process
    bkg = estimate_background(raw, P["bkg_block"])
    res = raw - bkg
    sigma = mad_sigma(res)
    stretch = make_stretch(res, sigma)
    img8 = stretch(res)

    # 2/3. detect + classify (on the FULL image, so tile borders never split decisions)
    lab, n, s_sm = detect(res, sigma)
    objs = ndi.find_objects(lab)

    # 4. pad + tile
    nr, nc = int(np.ceil(H / TILE)), int(np.ceil(W / TILE))
    PH, PW = nr * TILE, nc * TILE
    pad8 = np.full((PH, PW), stretch(np.zeros(1, np.float32))[0], np.uint8)
    pad8[:H, :W] = img8
    if save16:
        pad16 = np.full((PH, PW), int(np.median(raw)), np.uint16)
        pad16[:H, :W] = np.clip(raw, 0, 65535).astype(np.uint16)

    for d in ("images", "labels", "full", "images16"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    tile_lines = {(r, c): [] for r in range(nr) for c in range(nc)}
    catalog = []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        f = shape_features(m)
        cls = classify(f)
        y0, x0 = sl[0].start, sl[1].start
        amb = P["ambiguous"][0] <= f["elong"] <= P["ambiguous"][1]
        catalog.append(dict(image=stem, id=i, cls=CLASS_NAMES[cls], cx=x0 + m.shape[1] / 2,
                            cy=y0 + m.shape[0] / 2, review=int(amb),
                            **{k: round(v, 2) for k, v in f.items()}))
        for r in range(y0 // TILE, (y0 + m.shape[0] - 1) // TILE + 1):
            for c in range(x0 // TILE, (x0 + m.shape[1] - 1) // TILE + 1):
                ty0, tx0 = r * TILE, c * TILE
                sub = m[max(ty0 - y0, 0):TILE + ty0 - y0, max(tx0 - x0, 0):TILE + tx0 - x0]
                if sub.sum() < 3:
                    continue
                ox, oy = max(x0, tx0) - tx0, max(y0, ty0) - ty0
                for poly in polygons_for_tile(sub):
                    p = poly + [ox, oy]
                    p = np.clip(p / TILE, 0, 1)
                    tile_lines[(r, c)].append(
                        f"{cls} " + " ".join(f"{v:.6f}" for v in p.reshape(-1)))

    for (r, c), lines in tile_lines.items():
        name = f"{stem}_r{r}_c{c}"
        tile = pad8[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]
        cv2.imwrite(os.path.join(out, "images", name + ".png"), tile)
        if save16:
            cv2.imwrite(os.path.join(out, "images16", name + ".png"),
                        pad16[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE])
        with open(os.path.join(out, "labels", name + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))  # empty file = background tile

    # 5. repatch from the files on disk, prove lossless, crop padding, make overlay
    canvas = np.zeros((PH, PW), np.uint8)
    for r in range(nr):
        for c in range(nc):
            canvas[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = cv2.imread(
                os.path.join(out, "images", f"{stem}_r{r}_c{c}.png"), cv2.IMREAD_GRAYSCALE)
    full = canvas[:H, :W]
    assert np.array_equal(full, img8), "repatch mismatch - data lost!"
    cv2.imwrite(os.path.join(out, "full", f"{stem}_repatched.png"), full)

    ov = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)
    for rec in catalog:
        col = (255, 120, 0) if rec["cls"] == "star" else (0, 0, 255)
        rad = int(max(rec["major"], 6) / 2 + 5)
        cv2.rectangle(ov, (int(rec["cx"]) - rad, int(rec["cy"]) - rad),
                      (int(rec["cx"]) + rad, int(rec["cy"]) + rad), col, 1)
    cv2.imwrite(os.path.join(out, "full", f"{stem}_overlay.png"), ov)

    ns = sum(r["cls"] == "star" for r in catalog)
    print(f"{stem}: {H}x{W} -> {nr}x{nc} tiles | noise sigma={sigma:.1f} | "
          f"stars={ns} streaks={len(catalog) - ns} "
          f"review={sum(r['review'] for r in catalog)}")
    return catalog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="output")
    ap.add_argument("--save16", action="store_true", help="also write raw 16-bit tiles")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.input, "*.fit*")) +
                   glob.glob(os.path.join(a.input, "*.FIT*")))
    allrec = []
    for f in files:
        allrec += process(f, a.out, a.save16)
    if allrec:
        with open(os.path.join(a.out, "catalog.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(allrec[0].keys()))
            w.writeheader()
            w.writerows(allrec)
    with open(os.path.join(a.out, "data.yaml"), "w") as fh:
        fh.write("path: .\ntrain: images\nval: images\nnames:\n"
                 + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))


if __name__ == "__main__":
    main()
