"""Independent COLMAP mapper baseline on exactly the exported SIFT features/matches.

This is a controlled-input comparison, not COLMAP's default end-to-end pipeline.
Intrinsics are held fixed in both systems; inferred parameters are not ground truth.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pycolmap

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('sequence', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
data = json.loads(a.sequence.read_text())
a.output.mkdir(parents=True, exist_ok=True)
database_path = a.output / 'database.db'
if database_path.exists():
    p.error('baseline database already exists; use a new output directory')
width, height = data['metadata']['width'], data['metadata']['height']
views = {v['id']: v for v in data['views']}
frames = {f['id']: f for f in data['metadata']['frames']}
keypoints = {i: np.asarray(v['keypoints'], dtype=np.float64).reshape(-1, 2) + .5 for i, v in views.items()}
cameras, ids, verification = {}, {}, []
start = time.perf_counter()
with pycolmap.Database.open(database_path) as db:
    for i, v in views.items():
        k = v['camera']
        signature = json.dumps(k, sort_keys=True)
        if signature not in cameras:
            camera = pycolmap.Camera(model='OPENCV', width=width, height=height,
                                    params=[k['fx'], k['fy'], k['cx']+.5, k['cy']+.5, k.get('k1', 0), k.get('k2', 0), 0, 0])
            camera_id = db.write_camera(camera)
            camera.camera_id = camera_id
            sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_id)
            rig = pycolmap.Rig()
            rig.add_ref_sensor(sensor)
            rig_id = db.write_rig(rig)
            cameras[signature] = (camera, rig_id, sensor)
        camera, rig_id, sensor = cameras[signature]
        image_id = i+1
        frame = pycolmap.Frame(rig_id=rig_id)
        frame.add_data_id(pycolmap.data_t(sensor, image_id))
        frame_id = db.write_frame(frame)
        image = pycolmap.Image(image_id=image_id, name=frames[i]['name'], camera_id=camera.camera_id, frame_id=frame_id)
        db.write_image(image, use_image_id=True)
        db.write_keypoints(image_id, keypoints[i].astype(np.float32))
        ids[i] = (image_id, camera)
    options = pycolmap.TwoViewGeometryOptions()
    options.ransac.max_error = 3
    options.ransac.random_seed = 42
    options.ransac.num_threads = 1
    options.compute_relative_pose = True
    for pair in data['pairs']:
        i, j = pair['first'], pair['second']
        matches = np.asarray(pair['matches'], dtype=np.uint32).reshape(-1, 2)
        db.write_matches(ids[i][0], ids[j][0], matches)
        geometry = pycolmap.estimate_calibrated_two_view_geometry(ids[i][1], keypoints[i], ids[j][1], keypoints[j], matches, options)
        db.write_two_view_geometry(ids[i][0], ids[j][0], geometry)
        verification.append({'first': i, 'second': j, 'configuration': str(geometry.config), 'inliers': len(geometry.inlier_matches)})
options = pycolmap.IncrementalPipelineOptions()
options.random_seed = 42
options.num_threads = 1
options.extract_colors = False
options.ba_refine_focal_length = False
options.ba_refine_principal_point = False
options.ba_refine_extra_params = False
options.mapper.abs_pose_refine_focal_length = False
options.mapper.abs_pose_refine_extra_params = False
options.mapper.init_min_tri_angle = 1
options.mapper.init_min_num_inliers = 12
options.mapper.abs_pose_min_num_inliers = 12
options.mapper.abs_pose_max_error = 3
options.mapper.filter_max_reproj_error = 3
options.mapper.filter_min_tri_angle = .5
options.triangulation.min_angle = .5
options.triangulation.ignore_two_view_tracks = False
options.min_model_size = 2
options.min_num_matches = 12
models = pycolmap.incremental_mapping(database_path, a.sequence.parent / 'images', a.output / 'models', options)
results = []
for model_id, model in models.items():
    images = []
    for image in model.images.values():
        pose = image.cam_from_world()
        images.append({'id': image.image_id-1, 'name': image.name,
                       'pose': {'R': pose.rotation.matrix().reshape(-1).tolist(), 't': pose.translation.tolist()}})
    errors = []
    for point in model.points3D.values():
        for element in point.track.elements:
            image = model.images[element.image_id]
            camera = model.cameras[image.camera_id]
            pixel = camera.img_from_cam(image.cam_from_world() * point.xyz)
            observed = image.points2D[element.point2D_idx].xy
            errors.append(float(np.linalg.norm(pixel-observed)))
    results.append({'modelId': model_id, 'registeredViews': model.num_reg_images(), 'landmarks': model.num_points3D(),
                    'observations': model.compute_num_observations(), 'nativeMeanReprojectionErrorPx': model.compute_mean_reprojection_error(),
                    'meanEuclideanErrorPx': float(np.mean(errors)), 'rmsePx': float(np.sqrt(np.mean(np.square(errors)))),
                    'views': sorted(images, key=lambda v: v['id'])})
summary = {'pycolmapVersion': pycolmap.__version__, 'source': data['metadata'],
           'mode': 'same features/matches and fixed camera parameters; thresholds aligned where possible; remaining COLMAP options default',
           'options': options.todict(), 'verification': verification, 'models': results, 'elapsedSeconds': time.perf_counter()-start}
# pybind enum values need a string fallback in option snapshots.
(a.output / 'summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
print(json.dumps({'models': [{k: v for k, v in r.items() if k != 'views'} for r in results]}))
