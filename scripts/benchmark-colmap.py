"""Run two COLMAP references for a TypeCV eight-frame benchmark report.

The full mode runs SIFT extraction, exhaustive matching, and incremental mapping.
The controlled mode imports TypeCV's exact keypoints and putative matches, then
lets COLMAP estimate two-view geometry and run its incremental mapper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import time
from pathlib import Path

import numpy as np
import pycolmap


WIDTH, HEIGHT = 640, 360
FOCAL = 768.0
PRINCIPAL = (319.5, 179.5)  # TypeCV pixel-center convention; COLMAP adds 0.5.


def snapshot_model(model, id_offset=1):
    views = []
    errors = []
    for image in model.images.values():
        pose = image.cam_from_world()
        frame_match = re.fullmatch(r"frame-(\d+)\.jpg", image.name)
        view_id = int(frame_match.group(1)) if frame_match else image.image_id - id_offset
        views.append({
            "id": view_id,
            "name": image.name,
            "pose": {
                "R": pose.rotation.matrix().reshape(-1).tolist(),
                "t": pose.translation.tolist(),
            },
        })
        for point2d in image.points2D:
            if not point2d.has_point3D():
                continue
            point = model.points3D[point2d.point3D_id]
            camera = model.cameras[image.camera_id]
            pixel = camera.img_from_cam(image.cam_from_world() * point.xyz)
            errors.append(float(np.linalg.norm(pixel - point2d.xy)))
    return {
        "registeredViews": model.num_reg_images(),
        "landmarks": model.num_points3D(),
        "observations": model.compute_num_observations(),
        "nativeMeanReprojectionErrorPx": model.compute_mean_reprojection_error(),
        "meanEuclideanErrorPx": float(np.mean(errors)) if errors else None,
        "rmsePx": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        "views": sorted(views, key=lambda view: view["id"]),
    }


def compare_reconstruction(typecv_model, colmap_model):
    """Compare poses and reprojection on matched image observations.

    TypeCV uses integer-centered pixels. COLMAP's image coordinates and camera
    principal point are shifted by +0.5, so observations and projections are
    shifted back by 0.5 before the shared-pixel error comparison.
    """
    type_views = {view["id"]: view for view in typecv_model["views"]}
    colmap_views = {view["id"]: view for view in snapshot_model(colmap_model)["views"]}
    common_ids = sorted(type_views.keys() & colmap_views.keys())
    alignment = {"commonViewIds": common_ids, "registeredViews": len(common_ids)}
    if len(common_ids) >= 3:
        source_rotations = np.asarray(
            [type_views[i]["pose"]["R"] for i in common_ids], dtype=np.float64
        ).reshape(-1, 3, 3)
        reference_rotations = np.asarray(
            [colmap_views[i]["pose"]["R"] for i in common_ids], dtype=np.float64
        ).reshape(-1, 3, 3)
        source_centers = np.asarray([
            -rotation.T @ np.asarray(type_views[i]["pose"]["t"], dtype=np.float64)
            for i, rotation in zip(common_ids, source_rotations)
        ])
        reference_centers = np.asarray([
            -rotation.T @ np.asarray(colmap_views[i]["pose"]["t"], dtype=np.float64)
            for i, rotation in zip(common_ids, reference_rotations)
        ])
        u, _, vt = np.linalg.svd(
            sum(reference.T @ source for reference, source in zip(reference_rotations, source_rotations))
        )
        rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
        rotated = (rotation @ source_centers.T).T
        source_mean, reference_mean = rotated.mean(axis=0), reference_centers.mean(axis=0)
        variance = float(np.sum((rotated - source_mean) ** 2))
        scale = float(
            np.sum((rotated - source_mean) * (reference_centers - reference_mean)) / variance
        ) if variance > 1e-20 else float("nan")
        angle_errors = []
        for reference, source in zip(reference_rotations, source_rotations):
            delta = reference @ (source @ rotation.T).T
            angle_errors.append(float(np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1)))))
        alignment.update({
            "medianRotationDifferenceDeg": float(np.median(angle_errors)),
            "meanRotationDifferenceDeg": float(np.mean(angle_errors)),
            "scaleTypeCvToColmap": scale,
        })
        extent = float(np.max(np.linalg.norm(reference_centers - reference_mean, axis=1)))
        if scale > 0 and extent > 1e-12:
            aligned = scale * (rotated - source_mean) + reference_mean
            center_errors = np.linalg.norm(aligned - reference_centers, axis=1) / extent
            alignment.update({
                "status": "aligned",
                "medianCenterDifferenceNormalizedByReferenceRadius": float(np.median(center_errors)),
                "centerRmseNormalizedByReferenceRadius": float(np.sqrt(np.mean(center_errors**2))),
            })
        else:
            alignment.update({
                "status": "unavailable",
                "reason": "Similarity alignment did not produce a positive scale and measurable trajectory extent",
            })
    else:
        alignment.update({
            "status": "unavailable",
            "reason": "Fewer than three common registered views",
        })

    type_landmarks = {}
    for landmark in typecv_model["landmarks"]:
        for observation in landmark["observations"]:
            type_landmarks.setdefault(observation["viewId"], []).append((
                np.asarray(landmark["xyz"], dtype=np.float64),
                np.asarray(observation["pixel"], dtype=np.float64),
            ))
    colmap_observations = {}
    colmap_images = {}
    for image in colmap_model.images.values():
        frame_match = re.fullmatch(r"frame-(\d+)\.jpg", image.name)
        if not frame_match:
            continue
        view_id = int(frame_match.group(1))
        colmap_images[view_id] = image
        rows = []
        for point2d in image.points2D:
            if point2d.has_point3D():
                rows.append((
                    np.asarray(point2d.xy, dtype=np.float64) - 0.5,
                    np.asarray(colmap_model.points3D[point2d.point3D_id].xyz, dtype=np.float64),
                ))
        colmap_observations[view_id] = rows

    typecv_errors, colmap_errors = [], []
    shared_count = 0
    for view_id, observations in type_landmarks.items():
        image = colmap_images.get(view_id)
        candidates = colmap_observations.get(view_id, [])
        if image is None or not candidates:
            continue
        pose = image.cam_from_world()
        colmap_rotation = np.asarray(pose.rotation.matrix(), dtype=np.float64)
        colmap_translation = np.asarray(pose.translation, dtype=np.float64)
        camera = colmap_model.cameras[image.camera_id]
        type_view = type_views[view_id]
        type_camera = type_view["camera"]
        type_rotation = np.asarray(type_view["pose"]["R"], dtype=np.float64).reshape(3, 3)
        type_translation = np.asarray(type_view["pose"]["t"], dtype=np.float64)
        for type_xyz, pixel in observations:
            distances = [float(np.linalg.norm(colmap_pixel - pixel)) for colmap_pixel, _ in candidates]
            candidate_index = int(np.argmin(distances))
            if distances[candidate_index] > 1.0:
                continue
            colmap_xyz = candidates[candidate_index][1]
            type_camera_xyz = type_rotation @ type_xyz + type_translation
            colmap_camera_xyz = colmap_rotation @ colmap_xyz + colmap_translation
            if type_camera_xyz[2] <= 0 or colmap_camera_xyz[2] <= 0:
                continue
            type_projection = np.asarray([
                type_camera["fx"] * type_camera_xyz[0] / type_camera_xyz[2] + type_camera["cx"],
                type_camera["fy"] * type_camera_xyz[1] / type_camera_xyz[2] + type_camera["cy"],
            ])
            colmap_projection = np.asarray(camera.img_from_cam(colmap_camera_xyz)) - 0.5
            typecv_errors.append(float(np.linalg.norm(type_projection - pixel)))
            colmap_errors.append(float(np.linalg.norm(colmap_projection - pixel)))
            shared_count += 1

    def error_stats(values):
        if not values:
            return {"count": 0, "meanPx": None, "medianPx": None, "rmsePx": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "meanPx": float(np.mean(array)),
            "medianPx": float(np.median(array)),
            "rmsePx": float(np.sqrt(np.mean(array**2))),
        }

    return {
        "cameraAlignment": alignment,
        "sharedObservationReprojection": {
            "pixelConvention": "TypeCV integer-centered coordinates; COLMAP keypoints and projections shifted by -0.5px",
            "pixelAssociation": "nearest registered COLMAP observation within 1px of a TypeCV tracked pixel; both projections use the same TypeCV pixel",
            "sharedObservationCount": shared_count,
            "typecv": error_stats(typecv_errors),
            "colmap": error_stats(colmap_errors),
        },
    }


def configure_mapper():
    options = pycolmap.IncrementalPipelineOptions()
    options.random_seed = 42
    options.num_threads = 1
    options.extract_colors = False
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.mapper.abs_pose_refine_focal_length = False
    options.mapper.abs_pose_refine_extra_params = False
    options.mapper.init_min_tri_angle = 1.0
    options.mapper.init_min_num_inliers = 12
    options.mapper.abs_pose_min_num_inliers = 12
    options.mapper.abs_pose_max_error = 3.0
    options.mapper.filter_max_reproj_error = 3.0
    options.mapper.filter_min_tri_angle = 0.5
    options.triangulation.min_angle = 0.5
    options.triangulation.ignore_two_view_tracks = False
    options.min_model_size = 2
    options.min_num_matches = 12
    return options


def load_report(path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))
    data = report.get("source", report)
    comparison = data.get("comparisonInput")
    if not comparison or not comparison.get("images") or not comparison.get("pairs"):
        raise ValueError("benchmark JSON must contain source.comparisonInput.images and pairs")
    if not data.get("reconstruction", {}).get("views") or not data.get("reconstruction", {}).get("landmarks"):
        raise ValueError("benchmark JSON must contain source.reconstruction views and landmarks")
    images = sorted(comparison["images"], key=lambda image: image["id"])
    if len(images) != 8 or [image["id"] for image in images] != list(range(8)):
        raise ValueError("expected eight comparisonInput images with IDs 0 through 7")
    if report.get("refineSharedFocal") is not False:
        raise ValueError("TypeCV report must use --refine-focal=0 for a fixed-camera comparison")
    for image in images:
        camera = image["camera"]
        actual = [camera["fx"], camera["fy"], camera["cx"], camera["cy"]]
        expected = [FOCAL, FOCAL, PRINCIPAL[0], PRINCIPAL[1]]
        if any(abs(a - b) > 1e-6 for a, b in zip(actual, expected)):
            raise ValueError(
                f"image {image['id']} intrinsics {actual} differ from required fixed camera {expected}; "
                "rerun TypeCV with --refine-focal=0 and the 640x360 fixture"
            )
    for pair in comparison["pairs"]:
        if pair["first"] not in range(8) or pair["second"] not in range(8):
            raise ValueError(f"invalid comparison pair image IDs: {pair.get('first')}, {pair.get('second')}")
    return report, data, images, comparison["pairs"]


def hardware_info():
    cuda = getattr(pycolmap, "has_cuda", None)
    if callable(cuda):
        try:
            cuda = bool(cuda())
        except Exception as exc:  # capability introspection should not abort CPU runs
            cuda = f"unknown: {exc}"
    selected = "cuda" if cuda is True else "cpu" if cuda is False else "auto"
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logicalCpuCount": os.cpu_count(),
        "pycolmapVersion": getattr(pycolmap, "__version__", "unknown"),
        "pycolmapCudaCapability": cuda,
        "selectedFeatureDevice": selected,
        "note": "CUDA capability is reported when exposed by this pycolmap build; wall times are stage-specific.",
    }


def prepare_camera(db, image_reader_options=None):
    camera = pycolmap.Camera(
        model="OPENCV", width=WIDTH, height=HEIGHT,
        params=[FOCAL, FOCAL, PRINCIPAL[0] + 0.5, PRINCIPAL[1] + 0.5, 0, 0, 0, 0],
    )
    camera_id = db.write_camera(camera)
    camera.camera_id = camera_id
    return camera


def run_controlled(typecv_model, images_dir, output_dir, images, pairs):
    database_path = output_dir / "database.db"
    verification = []
    camera = None
    ids = {}
    keypoints = {
        image["id"]: np.asarray(image["keypoints"], dtype=np.float64).reshape(-1, 2) + 0.5
        for image in images
    }
    timings = {}
    stage_start = time.perf_counter()
    with pycolmap.Database.open(database_path) as db:
        camera = prepare_camera(db)
        sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera.camera_id)
        rig = pycolmap.Rig()
        rig.add_ref_sensor(sensor)
        rig_id = db.write_rig(rig)
        for item in images:
            image_id = item["id"] + 1
            frame = pycolmap.Frame(rig_id=rig_id)
            frame.add_data_id(pycolmap.data_t(sensor, image_id))
            frame_id = db.write_frame(frame)
            name = f"frame-{item['id']:02d}.jpg"
            colmap_image = pycolmap.Image(
                image_id=image_id, name=name, camera_id=camera.camera_id, frame_id=frame_id
            )
            db.write_image(colmap_image, use_image_id=True)
            db.write_keypoints(image_id, keypoints[item["id"]].astype(np.float32))
            ids[item["id"]] = image_id
        timings["databaseAndKeypointImportMs"] = (time.perf_counter() - stage_start) * 1000

        geometry_start = time.perf_counter()
        options = pycolmap.TwoViewGeometryOptions()
        options.ransac.max_error = 3.0
        options.ransac.random_seed = 42
        options.ransac.num_threads = 1
        options.compute_relative_pose = True
        for pair in pairs:
            i, j = pair["first"], pair["second"]
            matches = np.asarray(pair["matches"], dtype=np.uint32).reshape(-1, 2)
            db.write_matches(ids[i], ids[j], matches)
            geometry = pycolmap.estimate_calibrated_two_view_geometry(
                camera, keypoints[i], camera, keypoints[j], matches, options
            )
            db.write_two_view_geometry(ids[i], ids[j], geometry)
            verification.append({
                "first": i, "second": j, "configuration": str(geometry.config),
                "putativeMatches": int(len(matches)),
                "inliers": int(len(geometry.inlier_matches)),
            })
        timings["geometryVerificationMs"] = (time.perf_counter() - geometry_start) * 1000

    mapper_options = configure_mapper()
    mapper_start = time.perf_counter()
    models = pycolmap.incremental_mapping(
        database_path, images_dir, output_dir / "models", mapper_options
    )
    timings["incrementalMapperMs"] = (time.perf_counter() - mapper_start) * 1000
    results = []
    for model_id, model in models.items():
        item = snapshot_model(model)
        item["modelId"] = model_id
        item["comparisonAgainstTypeCV"] = compare_reconstruction(typecv_model, model)
        results.append(item)
        model_path = output_dir / f"model-{model_id}"
        model_path.mkdir()
        model.write(str(model_path))
    return {
        "mode": "controlled-typecv-keypoints-and-matches",
        "timings": timings,
        "verification": verification,
        "options": mapper_options.todict(),
        "models": results,
    }


def run_full(images_dir, output_dir, images, typecv_model):
    database_path = output_dir / "database.db"
    device = pycolmap.Device
    selected_device = device.cuda if getattr(pycolmap, "has_cuda", False) else device.cpu
    names = [f"frame-{image['id']:02d}.jpg" for image in images]
    timings = {}
    db_start = time.perf_counter()
    with pycolmap.Database.open(database_path) as db:
        # The website fixture's camera is fixed and shared across all eight images.
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "OPENCV"
        reader.camera_params = "768,768,320,180,0,0,0,0"
        camera_mode = pycolmap.CameraMode.SINGLE
        extraction = pycolmap.FeatureExtractionOptions()
        pycolmap.extract_features(
            database_path, images_dir, image_names=names, camera_mode=camera_mode,
            reader_options=reader, extraction_options=extraction,
            device=selected_device,
        )
    timings["featureExtractionMs"] = (time.perf_counter() - db_start) * 1000

    matching_start = time.perf_counter()
    matching = pycolmap.FeatureMatchingOptions()
    device = pycolmap.Device
    pycolmap.match_exhaustive(
        database_path, matching_options=matching,
        device=selected_device,
    )
    timings["exhaustiveMatchingAndVerificationMs"] = (time.perf_counter() - matching_start) * 1000

    mapper_options = configure_mapper()
    mapper_start = time.perf_counter()
    models = pycolmap.incremental_mapping(
        database_path, images_dir, output_dir / "models", mapper_options
    )
    timings["incrementalMapperMs"] = (time.perf_counter() - mapper_start) * 1000
    results = []
    for model_id, model in models.items():
        item = snapshot_model(model)
        item["modelId"] = model_id
        item["comparisonAgainstTypeCV"] = compare_reconstruction(typecv_model, model)
        results.append(item)
        model_path = output_dir / f"model-{model_id}"
        model_path.mkdir()
        model.write(str(model_path))
    return {
        "mode": "colmap-sift-exhaustive-matching-and-mapper",
        "timings": timings,
        "options": mapper_options.todict(),
        "models": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path, help="TypeCV e2e JSON with comparisonInput")
    parser.add_argument("output", type=Path, help="new output directory; must not exist")
    parser.add_argument("--mode", choices=("both", "full", "controlled"), default="both")
    parser.add_argument("--images", type=Path, default=Path("website/public/sample/pyramid"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output directory already exists; choose a fresh path: {args.output}")
    if not args.images.is_dir():
        parser.error(f"fixture image directory does not exist: {args.images}")
    report, data, images, pairs = load_report(args.benchmark)
    for image in images:
        path = args.images / f"frame-{image['id']:02d}.jpg"
        if not path.is_file():
            parser.error(f"missing fixture image: {path}")
    args.output.mkdir(parents=True)
    fixture_digest = hashlib.sha256()
    for image in images:
        fixture_digest.update((args.images / f"frame-{image['id']:02d}.jpg").read_bytes())

    outputs = {}
    modes = ("full", "controlled") if args.mode == "both" else (args.mode,)
    for mode in modes:
        output_dir = args.output / mode
        output_dir.mkdir()
        if mode == "full":
            outputs[mode] = run_full(args.images, output_dir, images, data["reconstruction"])
        else:
            outputs[mode] = run_controlled(
                data["reconstruction"], args.images, output_dir, images, pairs
            )
        ranked_models = sorted(
            outputs[mode]["models"], key=lambda model: model["registeredViews"], reverse=True
        )
        outputs[mode]["comparisonAgainstTypeCV"] = (
            ranked_models[0]["comparisonAgainstTypeCV"] if ranked_models else None
        )
    result = {
        "schemaVersion": 1,
        "benchmark": "colmap-reference-eight-frame-640x360",
        "input": str(args.benchmark.resolve()),
        "fixture": {"width": WIDTH, "height": HEIGHT, "frameCount": len(images), "sha256Concat": fixture_digest.hexdigest()},
        "camera": {"model": "OPENCV", "fx": FOCAL, "fy": FOCAL, "cxTypeCV": PRINCIPAL[0], "cyTypeCV": PRINCIPAL[1], "principalPointOffsetForColmap": 0.5, "refinement": "fixed"},
        "hardware": hardware_info(),
        "typecvWebGpu": {
            "backend": data.get("backend"),
            "adapter": data.get("adapter"),
            "requestedMode": report.get("mode"),
        },
        "typecv": {
            "featureMode": report.get("featureMode"),
            "maxFeatures": report.get("maxFeatures"),
            "smoothDescriptors": report.get("smoothDescriptors"),
            "matching": report.get("matching"),
            "reconstruction": data.get("reconstruction"),
        },
        "colmap": outputs,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({mode: {"timings": value["timings"], "models": [{key: item[key] for key in ("modelId", "registeredViews", "landmarks", "observations", "meanEuclideanErrorPx")} for item in value["models"]]} for mode, value in outputs.items()}))


if __name__ == "__main__":
    main()
