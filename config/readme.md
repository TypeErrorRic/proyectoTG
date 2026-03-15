# Segmentation Config Reference

`segmentar_defaults.json` is strict JSON and must contain data only (no inline comments).
Parameter documentation is maintained here.

Per-image specialized configs are stored in:
`config/segmentar_por_imagen/segmentar_defaults_<image_name>.json`

## groundParams

- `dist_thresh`: Maximum point-to-plane distance (meters) to count as an inlier for ground fitting.
- `max_iters`: Maximum number of RANSAC iterations for ground plane estimation.
- `min_inliers`: Minimum number of inlier points required to accept a ground plane.
- `subsample_stride`: Pixel stride used to subsample depth data before ground fitting.
- `up_axis`: Reference up direction used to validate plane orientation.
- `max_angle_deg`: Maximum allowed angle (degrees) between detected ground normal and expected orientation.
- `seed`: Random seed used by RANSAC sampling.
- `score_subset`: Number of sampled points used to score candidate ground planes.
- `orientation`: Expected plane type/orientation preset used by the ground detector.
- `early_stop_ratio`: Early-stop ratio of inliers to total candidates for ground RANSAC.
- `batch_size`: Number of plane hypotheses evaluated per RANSAC batch.
- `low_height_pct`: Lower height percentile used to bias candidate ground points.
- `roi_bottom_fraction`: Initial bottom image fraction used as ROI for ground candidates.
- `roi_expand_step`: Step size used to expand ROI upward when not enough candidates are found.
- `max_agg_points`: Maximum number of aggregated points used for ground RANSAC.
- `refine_full_res`: Whether to refine the detected ground plane using full-resolution data.
- `refine_max_points`: Maximum number of points used during ground refinement.
- `refine_dist_mult`: Distance-threshold multiplier used when collecting inliers for refinement.
- `ground_mask_refine`: Whether to apply post-processing refinement to the ground mask.

## wallParams

- `wall_subsample_stride`: Pixel stride used to subsample depth data before wall fitting.
- `wall_dist_thresh`: Maximum point-to-plane distance (meters) to count as a wall inlier.
- `wall_max_iters`: Maximum number of RANSAC iterations for wall plane estimation.
- `wall_min_inliers`: Minimum number of inlier points required to accept a wall plane.
- `wall_max_angle_deg`: Maximum allowed wall tilt angle (degrees) from expected vertical orientation.
- `wall_score_subset`: Number of sampled points used to score candidate wall planes.
- `wall_early_stop_ratio`: Early-stop ratio of inliers to total candidates for wall RANSAC.
- `wall_batch_size`: Number of wall hypotheses evaluated per RANSAC batch.
- `wall_refine_dist_mult`: Distance-threshold multiplier used when refining wall planes.
- `max_up_dot`: Maximum absolute dot product between wall normal and up axis.
- `ground_perp_deg`: Tolerance angle (degrees) for wall planes being perpendicular to ground.
- `wall_ortho_deg`: Tolerance angle (degrees) for orthogonality checks between wall planes.
- `wall_parallel_deg`: Tolerance angle (degrees) to classify wall planes as parallel.
- `wall_parallel_distance_m`: Maximum separation (meters) for considering parallel wall relations.
- `wall_mask_refine`: Whether to apply post-processing refinement to the wall mask.

## doorParams

- `door_hsv_enabled`: Enables/disables HSV color refinement for door segmentation.
- `door_hue_tol`: HSV hue tolerance used for door color filtering.
- `door_min_s`: Minimum HSV saturation accepted for door candidates.
- `door_min_v`: Minimum HSV value (brightness) accepted for door candidates.
- `door_glare_s_max`: Maximum HSV saturation used to identify glare regions.
- `door_glare_v_min`: Minimum HSV value used to identify glare regions.
- `door_glare_v_clip`: HSV value clamp applied to glare pixels.
- `door_ground_parallel_deg`: Maximum allowed door-plane tilt angle (degrees) relative to ground constraints.
- `door_plane_inlier_ratio`: Minimum inlier ratio required to accept a fitted door plane.

## wallParamsOverrides

- `max_angle_deg`: Override: maximum wall angle (degrees) used in shared plane fitting settings.
- `max_planes`: Maximum number of wall planes to extract.
- `enforce_vertical`: Whether to enforce vertical-orientation constraints during wall extraction.
- `max_up_dot`: Override: maximum wall normal alignment with the up axis.
- `refine`: Whether to run wall-plane refinement after initial detection.
- `ground_perp_deg`: Override: wall-to-ground perpendicular tolerance (degrees).
- `wall_ortho_deg`: Override: orthogonality tolerance (degrees) between wall planes.
- `wall_parallel_deg`: Override: parallelism tolerance (degrees) between wall planes.
- `wall_parallel_distance_m`: Override: distance threshold (meters) for parallel wall relationships.
