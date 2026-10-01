"""Immutable identities and parameters for the one-shot spatial-v1 PBR screen."""

FROZEN_INPUT_SHA256 = "b69e9d1853b6c3f207a4f7d737c1e4cbf1542b3c2b7574045901a4e85a7cd7b1"
FROZEN_MESH_SHA256 = "2d3a8b5ac0bfe44a60c8a7d83ff5b2042a6fac3b715806ac2e454f19b43f52d3"
FROZEN_SCENE_SHA256 = "a460ee742fefb0cd1f29c55bf8f4290de0bbe08556fca61daabb200c28e52dc2"
FROZEN_CORRESPONDENCE_SHA256 = "cfbfc146f65451137da58887ef969ec3c06b0fef4c893be29486cfd392933474"
FROZEN_CORRESPONDENCE_META_SHA256 = "02c5a2a01bae87fa5ce34f094e39c6b4ff5224f6ce3873bf700f52c25b30b821"
FROZEN_ESTIMATOR_SHA256 = "eb017f6297ba3958e7a08ccf901abf23fb98aafd6a170c6380e7595ee87f35e9"
FROZEN_INPUT_VALIDATION_SHA256 = "d8e633ec75133610d25618db6eaf3b15dda8cf3acaf1d26721ab8a0561434164"
FROZEN_FORWARD_MODEL_SHA256 = "bcb39942b4e62a416aabe9aed50ec03773028627872cd598ede90d9b2a31bd79"
FROZEN_SELECTION_SHA256 = "385c5c2623ded529aac5a92ca2674b86c5ee744948359145a86401fc34d148d7"
FROZEN_QUALITY_RENDERER_SHA256 = "be498cedc2e6c9c2b10a682ff6f04f3cedae2171dd672314755021994ae43117"
FROZEN_REGION_INPUT_SHA256 = "173ab9622f6df89e1f7ed228216f41467d6487a459ed238a7256aacaf358db6a"
FROZEN_PARAMETERS = {
    "resolution": 32,
    "max_nfev": 120,
    "min_samples": 3,
    "normal_spatial_weight": 0.15,
    "normal_geometry_weight": 0.015,
}
INPUT_FIELDS = {"training_observations", "training_view_masks"}
REGION_SCHEMA = "modly.ticket08.topology-material-regions.v1"
CANDIDATE_ID = "modly.project-owned-region-spatial-inverse-v1"
OUTPUT_NAME = "ticket08-region-spatial-v1-frozen-estimate.npz"
REGION_MAP_NAME = "ticket08-region-spatial-v1-frozen-region-map.json"
REPORT_NAME = "ticket08-region-spatial-v1-frozen-run.json"
SCORE_NAME = "ticket08-region-spatial-v1-frozen-quality-score.json"
