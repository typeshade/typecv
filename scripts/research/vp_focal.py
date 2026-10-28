#!/usr/bin/env python3
"""Offline feasibility probe: focal length from vanishing points (numpy + PIL only).

Usage: python3 scripts/research/vp_focal.py [--json out.json] [sample-name ...]
Pipeline: gradient-based line-support regions (LSD-like) -> segments ->
sequential RANSAC vanishing points -> f from orthogonal pairs
(square pixels, zero skew, principal point at the image centre).
"""
import json, sys, os, math
import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "fixtures", "lab-images")
rng = np.random.default_rng(1)


def load_gray(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float64)


def blur(a):
    k = np.array([1, 4, 6, 4, 1], float) / 16
    a = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, a)
    return np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 0, a)


def detect_segments(img, min_len=15, tol=math.radians(22.5), mag_thr=5.0):
    """Region growing on level-line angle, then PCA line fit per region."""
    g = blur(img)
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[1:-1, 1:-1] = (g[1:-1, 2:] - g[1:-1, :-2]) / 2
    gy[1:-1, 1:-1] = (g[2:, 1:-1] - g[:-2, 1:-1]) / 2
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gx, -gy)  # level-line angle
    H, W = g.shape
    ok = mag > mag_thr
    used = ~ok
    order = np.argsort(-mag, axis=None)
    order = order[: int(ok.sum())]
    segs = []
    for idx in order:
        y, x = divmod(int(idx), W)
        if used[y, x]:
            continue
        used[y, x] = True
        pts = [(y, x)]
        sx, sy = math.cos(ang[y, x]), math.sin(ang[y, x])
        a0 = ang[y, x]
        i = 0
        while i < len(pts):
            py, px = pts[i]; i += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < H and 0 <= nx < W and not used[ny, nx]:
                        d = abs(((ang[ny, nx] - a0 + math.pi) % (2 * math.pi)) - math.pi)
                        if d < tol:
                            used[ny, nx] = True
                            pts.append((ny, nx))
                            sx += math.cos(ang[ny, nx]); sy += math.sin(ang[ny, nx])
                            a0 = math.atan2(sy, sx)
        if len(pts) < min_len:
            continue
        P = np.array(pts, float)
        w = mag[P[:, 0].astype(int), P[:, 1].astype(int)]
        c = (P * w[:, None]).sum(0) / w.sum()
        Q = P - c
        cov = (Q * w[:, None]).T @ Q / w.sum()
        ev, evec = np.linalg.eigh(cov)
        if ev[1] <= 0 or math.sqrt(max(ev[0], 0) / ev[1]) > 0.18:
            continue
        d = evec[:, 1]
        t = Q @ d
        if t.max() - t.min() < min_len:
            continue
        p1 = c + d * t.min(); p2 = c + d * t.max()
        segs.append((p1[1], p1[0], p2[1], p2[0]))  # x1,y1,x2,y2
    return np.array(segs)


def seg_lines(segs, cx, cy):
    """Homogeneous lines in centred coords, plus midpoints and lengths."""
    p1 = np.c_[segs[:, 0] - cx, segs[:, 1] - cy, np.ones(len(segs))]
    p2 = np.c_[segs[:, 2] - cx, segs[:, 3] - cy, np.ones(len(segs))]
    lines = np.cross(p1, p2)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True) + 1e-12
    mid = (p1 + p2) / 2
    ln = np.hypot(p2[:, 0] - p1[:, 0], p2[:, 1] - p1[:, 1])
    return lines, mid, ln


def consistent(v, lines, mid, thr):
    """Segment direction vs direction midpoint->vp within thr (v homogeneous)."""
    d_seg = np.c_[-lines[:, 1], lines[:, 0]]
    if abs(v[2]) < 1e-9 * np.hypot(v[0], v[1]):
        d_vp = np.tile(v[:2] / np.hypot(v[0], v[1]), (len(lines), 1))
    else:
        d_vp = v[:2] / v[2] - mid[:, :2]
        d_vp = d_vp / (np.linalg.norm(d_vp, axis=1, keepdims=True) + 1e-12)
    return np.abs((d_seg * d_vp).sum(1)) > math.cos(thr)


def refine_vp(lines, w):
    A = lines * np.sqrt(w)[:, None]
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    return vt[-1]


def find_vps(lines, mid, ln, nvp=5, iters=800, thr=math.radians(2.0)):
    remaining = np.ones(len(lines), bool)
    vps = []
    for _ in range(nvp):
        idx = np.flatnonzero(remaining)
        if len(idx) < 6:
            break
        best, best_s = None, 0
        for _ in range(iters):
            a, b = rng.choice(idx, 2, replace=False)
            v = np.cross(lines[a], lines[b])
            n = np.linalg.norm(v)
            if n < 1e-9:
                continue
            v = v / n
            s = ln[idx][consistent(v, lines[idx], mid[idx], thr)].sum()
            if s > best_s:
                best, best_s = v, s
        if best is None:
            break
        for _ in range(3):
            m = consistent(best, lines, mid, thr) & remaining
            if m.sum() < 3:
                break
            best = refine_vp(lines[m], ln[m])
        m = consistent(best, lines, mid, thr) & remaining
        if m.sum() < 5:
            break
        vps.append({"v": best, "support": float(ln[m].sum()), "n": int(m.sum())})
        remaining &= ~m
    return vps


def f_from_pair(v1, v2):
    """f^2 = -(x1 x2 + y1 y2) / (w1 w2) in centred coordinates."""
    den = v1[2] * v2[2]
    if abs(den) < 1e-12:
        return None
    f2 = -(v1[0] * v2[0] + v1[1] * v2[1]) / den
    return math.sqrt(f2) if f2 > 0 else None


def estimate(path):
    img = load_gray(path)
    H, W = img.shape
    segs = detect_segments(img)
    out = {"segments": int(len(segs)), "f": None, "confidence": 0.0, "W": W}
    if len(segs) < 15:
        return out
    lines, mid, ln = seg_lines(segs, W / 2, H / 2)
    vps = find_vps(lines, mid, ln)
    total = ln.sum()
    best = None
    for i in range(len(vps)):
        for j in range(i + 1, len(vps)):
            f = f_from_pair(vps[i]["v"], vps[j]["v"])
            if f is None or not (0.25 * W <= f <= 4 * W):
                continue
            bal = min(vps[i]["support"], vps[j]["support"]) / total
            agree, fs = 0.0, [f]
            for k in range(len(vps)):
                if k in (i, j):
                    continue
                f1 = f_from_pair(vps[i]["v"], vps[k]["v"])
                f2 = f_from_pair(vps[j]["v"], vps[k]["v"])
                if f1 and f2 and abs(f1 - f) / f < 0.15 and abs(f2 - f) / f < 0.15:
                    agree = max(agree, min(vps[k]["support"] / total, 0.3))
                    fs = [f, f1, f2]
            score = bal + agree
            if best is None or score > best[0]:
                best = (score, f, bal, agree, float(np.median(fs)), len(fs) == 3)
    if best:
        score, f, bal, agree, fmed, tri = best
        out.update(f=fmed, f_pair=f, f_over_w=fmed / W, confidence=float(min(1.0, score * 3)),
                   balance=bal, triplet=bool(tri), vps=len(vps))
    return out


def main():
    args = sys.argv[1:]
    jpath = None
    if args and args[0] == "--json":
        jpath, args = args[1], args[2:]
    names = args or ["south-building", "gerrard-hall", "person-hall", "graham-hall", "pyramid"]
    res = {}
    for n in names:
        d = os.path.join(ROOT, n)
        meta = {}
        if os.path.exists(os.path.join(d, "sample.json")):
            meta = {fr["file"]: fr["camera"]["fx"] for fr in json.load(open(os.path.join(d, "sample.json")))["frames"]}
        rows = []
        for fn in sorted(x for x in os.listdir(d) if x.startswith("frame-") and x.endswith(".jpg")):
            r = estimate(os.path.join(d, fn))
            r["file"] = fn
            if fn in meta:
                r["f_true"] = meta[fn]
                if r["f"]:
                    r["err_pct"] = 100 * (r["f"] - meta[fn]) / meta[fn]
            rows.append(r)
            print(n, fn, r['segments'], None if r['f'] is None else round(r['f']), round(r['confidence'], 2), r.get('triplet'), None if 'err_pct' not in r else round(r['err_pct'], 1), flush=True)
        res[n] = rows
    if jpath:
        json.dump(res, open(jpath, "w"), indent=1)


if __name__ == "__main__":
    main()
