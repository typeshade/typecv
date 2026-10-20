"""Compare the KLT reference of this repository with OpenCV (docs/klt-tracking.md 7.3).

Offline record, not a CI gate and not a dependency. It needs python 3 with numpy and opencv
(`cv2.calcOpticalFlowPyrLK`, `cv2.goodFeaturesToTrack`), node, and `npm run build` before it runs.
`skimage.registration.optical_flow_ilk` is used when scikit-image is importable; otherwise that
part is "not run".

  python3 scripts/klt-opencv-compare.py synthetic [dx dy]
  python3 scripts/klt-opencv-compare.py video <file> <frame a> <frame b>

`video` decodes the file at 4 frames per second, 640 px wide, gray, like the lab, and takes two
sampled frames. Points are OpenCV's `goodFeaturesToTrack` (1024, quality 0.01, distance 8, block 3).
Both trackers start from the same points. OpenCV runs with the window 21 x 21, 3 levels above the
base (`maxLevel` 3), and 30 iterations or 0.01 px. Output: one JSON object on the last line.
"""
import json, os, subprocess, sys, tempfile
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELPER = os.path.join(ROOT, "scripts", "klt-opencv-reference.mjs")
LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01))


def node(*args):
    subprocess.run(["node", HELPER, *map(str, args)], check=True, cwd=ROOT)


def video_frames(path, a, b, width=640):
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=width,height", "-of", "csv=p=0", path]).decode().strip().split(",")
    w0, h0 = int(probe[0]), int(probe[1])
    w, h = width, round(h0 * width / w0)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps=4,scale={w}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], check=True, stdout=subprocess.PIPE).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w)
    return frames[a], frames[b], w, h


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    truth = None
    with tempfile.TemporaryDirectory() as dir:
        if mode == "synthetic":
            dx, dy = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (3.7, -2.1)
            w, h = 640, 360
            node("synthetic", dir, w, h, dx, dy)
            a = np.fromfile(f"{dir}/a.gray", np.uint8).reshape(h, w)
            b = np.fromfile(f"{dir}/b.gray", np.uint8).reshape(h, w)
            truth = (dx, dy)
            label = f"synthetic shift ({dx}, {dy}) px"
        else:
            a, b, w, h = video_frames(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
            a.tofile(f"{dir}/a.gray"); b.tofile(f"{dir}/b.gray")
            label = f"{os.path.basename(sys.argv[2])} sampled frames {sys.argv[3]} and {sys.argv[4]}"
        corners = cv2.goodFeaturesToTrack(a, maxCorners=1024, qualityLevel=0.01, minDistance=8, blockSize=3)
        pts = corners.reshape(-1, 2).astype(np.float32)
        json.dump(pts.ravel().tolist(), open(f"{dir}/points.json", "w"))
        node("track", dir, w, h)
        ref = json.load(open(f"{dir}/reference.json"))
        cv_next, st, err = cv2.calcOpticalFlowPyrLK(a, b, pts, None, **LK)
        cv_back, st_b, _ = cv2.calcOpticalFlowPyrLK(b, a, cv_next, None, **LK)
        fb = np.linalg.norm(cv_back - pts, axis=1)
        cv_ok = st.ravel() == 1
        cv_ok_fb = cv_ok & (st_b.ravel() == 1) & (fb < 0.5)
        ours = np.stack([ref["x"], ref["y"]], axis=1).astype(np.float64)
        ours_ok = np.array(ref["alive"], bool)
        both = cv_ok & ours_ok
        both_fb = cv_ok_fb & ours_ok
        d = np.linalg.norm(cv_next - ours, axis=1)

        def stats(mask):
            v = d[mask]
            return {"n": int(mask.sum()), "median": float(np.median(v)) if len(v) else None,
                    "p95": float(np.percentile(v, 95)) if len(v) else None,
                    "share_over_0.5px": float((v > 0.5).mean()) if len(v) else None}
        out = {"input": label, "points": int(len(pts)),
               "opencv_version": cv2.__version__,
               "kept": {"opencv_status": int(cv_ok.sum()), "opencv_status_and_fb05": int(cv_ok_fb.sum()),
                        "reference": int(ours_ok.sum()), "both": int(both.sum())},
               "distance_both": stats(both), "distance_both_opencv_fb05": stats(both_fb)}
        if truth:
            t = np.array(truth)
            out["error_vs_truth_median"] = {
                "opencv": float(np.median(np.linalg.norm(cv_next[both] - pts[both] - t, axis=1))),
                "reference": float(np.median(np.linalg.norm(ours[both] - pts[both] - t, axis=1)))}
        picks = np.array(ref["picks"], np.float64).reshape(-1, 2)
        if len(picks):
            near = lambda p, q: np.min(np.linalg.norm(q[None, :, :] - p[:, None, :], axis=2), axis=1)
            out["selection"] = {"opencv_corners": int(len(pts)), "reference_picks": int(len(picks)),
                                "opencv_with_pick_within_2px": float((near(pts, picks) <= 2).mean()),
                                "picks_with_opencv_corner_within_2px": float((near(picks, pts) <= 2).mean())}
        try:
            from skimage.registration import optical_flow_ilk
            flow = optical_flow_ilk(a.astype(np.float64) / 255, b.astype(np.float64) / 255, radius=7)
            v, u = flow  # (row, column) displacement of the pixels of a
            xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1); ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
            ilk = np.stack([pts[:, 0] + u[ys, xs], pts[:, 1] + v[ys, xs]], axis=1)
            out["skimage_ilk_distance_to_reference_both"] = float(np.median(np.linalg.norm(ilk[both] - ours[both], axis=1)))
        except ImportError:
            out["skimage_ilk"] = "not run: scikit-image is not installed"
        print(json.dumps(out))


main()
