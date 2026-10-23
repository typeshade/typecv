Dataset: graham-hall, COLMAP demo datasets (https://demuc.de/colmap/datasets/).
Derived 1280 x 853 JPEG previews of 12 images by name after the first 6, with the bundled COLMAP intrinsics and poses in sample.json, for an in-browser SfM demonstration. The original images and database are not included.
colmap-sift.bin holds the SIFT keypoints of these images that take part in a raw match in the dataset database, scaled to the preview size, and those matches, so the lab can run its mapper on COLMAP's own features.
