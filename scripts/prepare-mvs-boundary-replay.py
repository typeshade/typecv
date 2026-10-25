"""Prepare fixed-pose crops from exported depth archives, without committing source pixels.

Requires Python 3, numpy, Pillow and scipy. ROI coordinates below are manually annotated
benchmark regions for the three supplied records, never a production sky/foreground classifier.
"""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def rect(x0, y0, x1, y1):
    mask = np.zeros((720, 1280), bool)
    mask[y0:y1, x0:x1] = True
    return mask


def input_frames(directory, ids):
    # Stream the archives; depth arrays are not loaded. Never extract an arbitrary tar path.
    inputs = {}
    for archive in sorted(directory.glob("typecv-depths-c000-*.tar.gz")):
        with tarfile.open(archive, "r|gz") as stream:
            for member in stream:
                parts = member.name.split("/")
                if (len(parts) == 2 and parts[0] in {f"view-{i}" for i in ids}
                        and (parts[1] == "input.json" or parts[1].startswith("input.rgba8."))):
                    if not member.isfile() or member.size > 50 * 1024**2:
                        raise ValueError("Invalid input archive entry")
                    inputs[member.name] = stream.extractfile(member).read()
    result = {}
    for vid in ids:
        meta = json.loads(inputs[f"view-{vid}/input.json"])
        chunks = sorted(meta["parts"], key=lambda part: part["offset"])
        offset = 0
        data = []
        for chunk in chunks:
            value = inputs[chunk["path"]]
            if chunk["offset"] != offset or len(value) != chunk["bytes"]:
                raise ValueError("Input segments are incomplete")
            data.append(value)
            offset += len(value)
        pixels = np.frombuffer(b"".join(data), "u1").reshape(meta["height"], meta["width"], 4)
        if pixels.shape != (720, 1280, 4):
            raise ValueError("This benchmark's hand-annotated ROIs require 1280 x 720 inputs")
        result[vid] = Image.fromarray(pixels)
    return result


def roi_masks(rid, reference):
    if rid == "6c42e7d3":
        rgb = np.array(reference)[:, :, :3]
        gray = rgb @ np.array([.299, .587, .114])
        roi = rect(620, 8, 730, 105)
        metal = roi & (gray < 160)
        return {
            "sky": rect(20, 10, 470, 95) | rect(810, 10, 1260, 95),
            "tank": rect(600, 365, 700, 440),
            "metal": metal,
            "skyNearPole": roi & ~binary_dilation(metal, iterations=3)
                & (distance_transform_edt(~metal) <= 12),
        }
    if rid == "a4d735ce":
        pool = Image.new("1", (1280, 720))
        ImageDraw.Draw(pool).polygon([(618, 332), (657, 314), (714, 316), (754, 336),
                                     (741, 371), (705, 397), (651, 385), (623, 368)], fill=1)
        return {"leftRock": rect(80, 200, 350, 430), "rightRock": rect(1000, 300, 1150, 500),
                "pool": np.array(pool), "channel": rect(580, 530, 600, 550)}
    return {"sky": rect(20, 20, 350, 180) | rect(1000, 20, 1250, 180),
            "leftFar": rect(20, 265, 300, 460), "rightFar": rect(1010, 290, 1250, 460),
            "centralRock": rect(520, 200, 780, 450)}


def prepare(records, output):
    cases = []
    output.mkdir(parents=True, exist_ok=True)
    for rid, reference in [("6c42e7d3", 0), ("a4d735ce", 105), ("1e118cc5", 76)]:
        directory = records / f"record-{rid}"
        scene_bytes = (directory / "typecv-reconstruction.json").read_bytes()
        scene = json.loads(scene_bytes)
        manifest = json.loads((directory / "typecv-depths-c000.json").read_bytes())
        if not manifest["complete"] or manifest["missing"]:
            raise ValueError("Incomplete source report")
        row = next(m for m in manifest["maps"]
                   if m["viewId"] == reference and m["stage"] == "photometric")
        ids = {reference, *row["sourceViewIds"]}
        frames = input_frames(directory, ids)
        masks = roi_masks(rid, frames[reference])
        for kind in (["overview", "pole-native"] if rid == "6c42e7d3" else ["overview"]):
            dest = output / f"{rid}-{kind}"
            dest.mkdir(exist_ok=True)
            scale = .25 if kind == "overview" else 1
            x0, y0 = (520, 0) if kind == "pole-native" else (0, 0)
            width, height = (320, 180) if kind == "overview" else (320, 320)
            views, hashes = [], {}
            for view in scene["views"]:
                if view["id"] not in ids:
                    continue
                camera = view["camera"]
                camera = dict(camera, fx=camera["fx"] * scale, fy=camera["fy"] * scale,
                              cx=(camera["cx"] + .5) * scale - .5 - x0,
                              cy=(camera["cy"] + .5) * scale - .5 - y0)
                views.append(dict(view, camera=camera, width=width, height=height))
                image = frames[view["id"]]
                # Match the benchmark's frozen RGB->RGBA conversion, including Pillow's RGB
                # bilinear interpolation. All original alpha bytes must be opaque.
                if np.any(np.array(image)[:, :, 3] != 255):
                    raise ValueError("Transparent input needs an explicit replay protocol")
                image = image.convert("RGB")
                image = (image.resize((320, 180), Image.Resampling.BILINEAR)
                         if kind == "overview" else image.crop((x0, y0, x0 + width, y0 + height)))
                data = image.convert("RGBA").tobytes()
                (dest / f'{view["id"]}.rgba').write_bytes(data)
                hashes[str(view["id"])] = sha256(data)
            landmarks = []
            for landmark in scene["landmarks"]:
                observations = [dict(obs, pixel=[(obs["pixel"][0] + .5) * scale - .5 - x0,
                                                (obs["pixel"][1] + .5) * scale - .5 - y0])
                                for obs in landmark["observations"] if obs["viewId"] in ids]
                if observations:
                    landmarks.append({"xyz": landmark["xyz"], "observations": observations})
            encoded = json.dumps({"views": views, "landmarks": landmarks}).encode()
            (dest / "scene.json").write_bytes(encoded)
            item = {"name": dest.name, "record": rid, "reference": reference,
                    "width": width, "height": height,
                    "transform": {"scale": scale, "cropX": x0, "cropY": y0},
                    "sourceViewIds": sorted(ids - {reference}), "imageHashes": hashes,
                    "sceneHash": sha256(encoded), "sourceSceneHash": sha256(scene_bytes),
                    "masks": [], "maskHashes": {}}
            for name, mask in masks.items():
                image = Image.fromarray(np.uint8(mask)).resize(
                    (round(1280 * scale), round(720 * scale)), Image.Resampling.NEAREST)
                data = image.crop((x0, y0, x0 + width, y0 + height)).tobytes()
                if any(data):
                    (dest / f"mask-{name}.u8").write_bytes(data)
                    item["masks"].append(name)
                    item["maskHashes"][name] = sha256(data)
            (dest / "case.json").write_text(json.dumps(item, indent=2))
            cases.append(item)
    (output / "cases.json").write_text(json.dumps(cases, indent=2))
    print(json.dumps({"preparedCases": [case["name"] for case in cases]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True,
                        help="Directory containing record-<short-id> folders with exported files")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.records.resolve(), args.out.resolve())
