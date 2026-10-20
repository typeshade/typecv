"""Judge the tracker against the descriptors (docs/klt-tracking.md, step 6 record).

Reads the file of `scripts/measure-klt-agreement.mjs`. For every pair of keyframes it fits a
fundamental matrix with RANSAC on the matches that the tracker and the descriptors both found (the
agreed set), and measures every other match against it by the Sampson distance in pixels.

  python3 scripts/analyze-klt-agreement.py agreement.json

Prints one JSON object. Two questions:
  conflicts  A tracker match and a descriptor match that share one keypoint and differ on the
             other. Which side is nearer the epipolar geometry, and how far apart the two candidate
             keypoints are (the same place twice, as SIFT makes one keypoint for each orientation?).
  outliers   The share of tracker matches more than 1 px and more than 3 px from the epipolar
             line, against the same share of descriptor matches. The mapper verifies a pair at 3 px.
"""
import json, sys
import numpy as np
import cv2


def sampson(F, p, q):
    x1 = np.c_[p, np.ones(len(p))]; x2 = np.c_[q, np.ones(len(q))]
    Fx1 = x1 @ F.T; Ftx2 = x2 @ F
    num = np.sum(x2 * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def main():
    data = json.load(open(sys.argv[1]))
    pts = {k["id"]: np.array(k["points"], float).reshape(-1, 2) for k in data["keyframes"]}
    desc = {(p["first"], p["second"]): np.array(p["matches"], int).reshape(-1, 2) for p in data["descriptorPairs"]}
    trk = {(p["first"], p["second"]): np.array(p["matches"], int).reshape(-1, 2) for p in data["trackerPairs"]}
    total = dict(pairs=0, tracker=0, agreed=0, tracker_only=0, conflicts=0, descriptor_only=0)
    side = dict(tracker_nearer=0, descriptor_nearer=0, tracker_within_1px=0, descriptor_within_1px=0,
                only_tracker_within_1px=0, only_descriptor_within_1px=0, neither_within_1px=0, both_within_1px=0)
    gaps, err_t_conf, err_d_conf = [], [], []
    e_trk, e_desc, e_new, e_desc_only = [], [], [], []
    skipped = 0
    for key, T in trk.items():
        D = desc.get(key)
        if D is None or len(T) == 0:
            continue
        P, Q = pts[key[0]], pts[key[1]]
        dset = {(a, b) for a, b in D}
        dfwd = {a: b for a, b in D}; dbwd = {b: a for a, b in D}
        agreed = np.array([m for m in T if (m[0], m[1]) in dset])
        if len(agreed) < 16:
            skipped += 1
            continue
        F, mask = cv2.findFundamentalMat(P[agreed[:, 0]], Q[agreed[:, 1]], cv2.FM_RANSAC, 1.0, 0.999, 2000)
        if F is None or F.shape != (3, 3):
            skipped += 1
            continue
        total["pairs"] += 1
        total["tracker"] += len(T); total["agreed"] += len(agreed)
        tset = {(a, b) for a, b in T}
        conf_t, conf_d = [], []
        for a, b in T:
            if (a, b) in dset:
                continue
            if a in dfwd or b in dbwd:
                # The descriptor match that the tracker match disagrees with.
                bd = dfwd.get(a); ad = dbwd.get(b)
                if bd is not None:
                    conf_t.append((a, b)); conf_d.append((a, bd))
                else:
                    conf_t.append((a, b)); conf_d.append((ad, b))
                total["conflicts"] += 1
            else:
                total["tracker_only"] += 1
                e_new.append(sampson(F, P[[a]], Q[[b]])[0])
        total["descriptor_only"] += sum(1 for m in D if (m[0], m[1]) not in tset and m[0] not in {x for x, _ in T} and m[1] not in {y for _, y in T})
        e_trk.extend(sampson(F, P[T[:, 0]], Q[T[:, 1]]))
        e_desc.extend(sampson(F, P[D[:, 0]], Q[D[:, 1]]))
        for m in D:
            if (m[0], m[1]) not in tset and m[0] not in set(T[:, 0]) and m[1] not in set(T[:, 1]):
                e_desc_only.append(sampson(F, P[[m[0]]], Q[[m[1]]])[0])
        if conf_t:
            ct = np.array(conf_t); cd = np.array(conf_d)
            et = sampson(F, P[ct[:, 0]], Q[ct[:, 1]]); ed = sampson(F, P[cd[:, 0]], Q[cd[:, 1]])
            err_t_conf.extend(et); err_d_conf.extend(ed)
            side["tracker_nearer"] += int((et < ed).sum()); side["descriptor_nearer"] += int((ed < et).sum())
            side["tracker_within_1px"] += int((et < 1).sum()); side["descriptor_within_1px"] += int((ed < 1).sum())
            side["both_within_1px"] += int(((et < 1) & (ed < 1)).sum())
            side["only_tracker_within_1px"] += int(((et < 1) & (ed >= 1)).sum())
            side["only_descriptor_within_1px"] += int(((ed < 1) & (et >= 1)).sum())
            side["neither_within_1px"] += int(((et >= 1) & (ed >= 1)).sum())
            gaps.extend(np.linalg.norm(P[ct[:, 0]] - P[cd[:, 0]], axis=1) + np.linalg.norm(Q[ct[:, 1]] - Q[cd[:, 1]], axis=1))
    share = lambda v, t: float((np.array(v) > t).mean()) if len(v) else None
    med = lambda v: float(np.median(v)) if len(v) else None
    out = dict(input=sys.argv[1], pairs_skipped=skipped, totals=total, conflict_sides=side,
               conflict_candidate_gap_px=dict(median=med(gaps), share_under_2px=float((np.array(gaps) < 2).mean()) if gaps else None),
               conflict_median_epipolar_px=dict(tracker=med(err_t_conf), descriptor=med(err_d_conf)),
               tracker_matches=dict(n=len(e_trk), median_px=med(e_trk), over_1px=share(e_trk, 1), over_3px=share(e_trk, 3)),
               descriptor_matches=dict(n=len(e_desc), median_px=med(e_desc), over_1px=share(e_desc, 1), over_3px=share(e_desc, 3)),
               tracker_only_matches=dict(n=len(e_new), median_px=med(e_new), over_1px=share(e_new, 1), over_3px=share(e_new, 3)),
               descriptor_only_matches=dict(n=len(e_desc_only), median_px=med(e_desc_only), over_1px=share(e_desc_only, 1), over_3px=share(e_desc_only, 3)))
    print(json.dumps(out, indent=1))


main()
