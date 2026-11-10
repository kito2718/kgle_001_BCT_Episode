import os


class BaseConfig:
    run = "run79"

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "data")
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    vendor_src = os.path.join(project_root, "vendor", "repo", "src")
    vendor_scripts = os.path.join(project_root, "vendor", "repo", "scripts")
    weights_dir = os.path.join(project_root, "weights")
    models_dir = weights_dir

    output_dir = os.path.join(project_root, "outputs", run)
    predict_dir = os.path.join(output_dir, "predictions")
    results_dir = os.path.join(output_dir, "results")
    submission_csv = os.path.join(output_dir, "submission.csv")

    seed = 42
    device = "cuda"
    split = "test"
    clip_ids = None
    valid_ids = [
        "44b6_0113de3b",
        "44b6_0b24845f",
        "6bba_05b6850b",
        "6bba_05db0fb1",
    ]

    wandb_mode = "online" if os.environ.get("WANDB") == "1" else "disabled"
    wandb_project = "ctdd"
    wandb_entity = None
    wandb_name = None

    # Shared train/inference input size. Competition volumes are 64x256x256;
    # spatial downsampling is derived so every pipeline uses the same resolution.
    raw_volume_shape = [64, 256, 256]
    volume_shape = [64, 64, 64]
    time_length = 2
    q_low_key = "0.001"
    q_high_key = "0.999"

    unet_out_channels = 32
    unet_layers = (64, 128, 256)
    hidden_dim = 128
    n_heads = 4
    n_blocks = 4
    transformer_dropout = 0.1
    pos_per_dim = 8
    identity_dim = 16
    gradient_checkpointing = False
    pool_kernel_um = 5.0


# Baseline inference used by in-training competition evaluation.
class InferConfig(BaseConfig):
    checkpoint_paths = [
        "run53/fold_0/epoch_035.pth",
        # "run53/fold_1/epoch_040.pth",
    ]
    use_amp = True

    # Fast inference is shared by local prediction, submission and training validation.
    inference_tta_batch_size = 8
    inference_oom_fallback = True
    # None: all visible GPUs for device="cuda", or only the explicitly selected
    # device (e.g. "cuda:1"/"cpu"). Set ["cuda:0"] to reserve the other GPUs.
    inference_devices = None

    point_threshold = 0.90
    temporal_detection_fusion = False
    use_tta = False
    tta_mode = "8fixed"
    tta_head_fusion = False
    tta_idsim_fusion = False
    parabolic_refinement = True
    edge_threshold = 0.10  # candidate edges; ILP1/ILP2 admit only edges above their own threshold

    ilp_edge_weight = -1.0
    ilp_appearance_weight = 0.5
    ilp_appearance_scale = 0.5
    ilp_disappearance_weight = 1.8
    ilp_division_weight = 0.75
    ilp_division_scale = 4.0
    # Reward high-confidence detections and penalize low-confidence ones.
    ilp_node_weight = 8.0
    ilp_node_bias = 0.96
    ilp_solver = "highs"  # "highs": LP relaxation + exact MIP around fractional variables; "scip": full exact MIP
    ilp_relax_hops = 100  # constraint-graph radius freed around fractional LP variables, about one frame per hop

    idsim = {
        "enable": True,
        "weight": 0.4,
    }
    centroid = {
        "enable": False,
        "blend": 0.5,
        "rz": 2,
        "ryx": 5,
    }
    gap_fill = {
        "gate_um": 10.0,
        "velocity_residual_um": 2.5,
        "occupied_um": 3.0,
    }
    # Shift the t+1 coordinates handed to the link head back by the whole-frame motion
    # (detection-only local_flow median >= gate_um) so a drift frame looks like a still
    # frame to it. Output coordinates are untouched. fold0 adjJ +0.0060 at gate 2 um
    # (+0.0050 at 3 um, +0.0031 at 6 um), but an LB probe found no >=6 um drift frames
    # in the hidden test set, so it is off by default.
    drift_compensate = {
        "enable": False,
        "gate_um": 2.0,
        "neighbours": 15,         # matched pairs per node for the regional flow
        "drift_search_um": 12.0,  # whole-frame shift search range
        "coarse_sample": 200,     # nodes scored in the coarse shift search
    }
    track_postprocess = {
        "mode": "no_new_divisions",  # "none", "no_new_divisions" or "anchor_paths"
        "low_point_threshold": 0.6,
    }

    smooth_window = 4
    smooth_weight = 0.8
    smooth_preserve_motion = True
    smooth_max_shift_um = 5.0  # Cap how far smoothing moves a node, in um (0 disables).
    smooth_rts = {
        "enable": False,
        "model": "random_walk",
        "q": 0.06,         # Process noise variance per frame, in um².
        "share": 0.4,      # RTS share; the remainder uses local line smoothing.
        "robust_iter": 2,  # Reweight outlying observations after each fit.
        "z_noise": 2.0,    # Measurement variance multiplier for the z axis.
    }

    # Three ILP passes build and refine tracks (inference.solve_graph).
    # ILP1 solves the point_threshold detections as a reference; its forks and edges prune ILP2 (filter_ilp2)
    # and ILP3 (filter_ilp3). enable False: no ILP1, no pruning, only low_point_threshold detections are scored
    ilp1 = {"enable": True, "filter_ilp2": True, "filter_ilp3": True}
    ilp_edge_threshold = 0.35  # ILP1 (reference) admission; top of the ILP3 keep_bonus band
    ilp2 = {"edge_threshold": 0.30, "idsim": 0.5, "appearance_weight": 0.75}
    # ILP3 edge income p ** gamma: <= weak_lo unused, (weak_lo, keep_lo] scaffold (+bonus, removed
    # after the solve), (keep_lo, ilp_edge_threshold] +keep_bonus, (ilp_edge_threshold, mid_hi]
    # -mid_penalty; merged fork edges are forced
    ilp3 = {"weak_lo": 0.192, "keep_lo": 0.2894, "gamma": 0.8, "scaffold_bonus": 0.02,
            "keep_bonus": -0.1, "mid_hi": 0.6238, "mid_penalty": 0.1748, "idsim": 0.9941,
            "disappearance": 2.0625, "edge_weight": -0.9559}
    # Optional track repairs after ILP3, in this order (utils.py):
    # rematch: ILP3 components where dropping a chosen link loses < tau are matched again with idsim
    # and appearance scale raised to idsim / appearance_scale; new links need edge_prob > keep_p
    rematch = {"enable": False, "tau": 2.0, "idsim": 3.0, "appearance_scale": 1.0, "keep_p": 0.3}
    # gap_close: ILP4, track ends joined to track starts gaps[0]..gaps[1] frames later, missing nodes
    # interpolated or snapped to free detections; see utils.close_gaps for the cost
    gap_close = {"enable": False, "gaps": (2, 5), "max_cost": 5.5, "gap_cost": 0.5, "snap_bonus": 0.5,
                 "dist_um": 12.0, "snap_um": 3.5, "occupied_um": 3.0, "min_len": 8, "min_q": 0.85}
    # adjacent_join: track ends joined to track starts one frame later by a candidate edge with
    # p_lo < edge_prob <= p_hi and id_dissim < max_dissim
    adjacent_join = {"enable": False, "p_lo": 0.1, "p_hi": 0.3, "max_dissim": 0.35, "dist_um": 16.0}

    max_distance = 7.0
    adjustment_alpha = 0.1
    division_weight = 0.1


class SubmitConfig(InferConfig):
    # C1 (LB 0.971, 2026-09-26): equal 11-model ensemble; ILP1 -> ILP2 (ship admission) -> ILP3 relink
    # (forks and nodes fixed, (.2, .3] scaffold, edge income p, idsim .7, disappearance 2.0) ->
    # smoothing capped at 5 um + gap fill
    checkpoint_paths = [
        "run53/fold_0/epoch_035.pth",
        "run53/fold_1/epoch_025.pth",
        "run53/fold_2/epoch_020.pth",
        "run53/fold_3/epoch_028.pth",
        "run53/fold_4/epoch_028.pth",
        "run74/fold_0/epoch_027.pth",
        "run74/fold_1/epoch_030.pth",
        "run74/fold_2/epoch_024.pth",
        "run74/fold_3/epoch_025.pth",
        "run74/fold_4/epoch_025.pth",
        "run79/fold_0/epoch_024.pth",
    ]
    edge_threshold = 0.20
    # group N (2026-09-29) on the ILP1/rematch/adjacent-off base: ilp2.idsim .4->.5, division weight .75->.85, ILP3 keep_lo
    # .3->.25, gamma 1->.5, disappearance 2->3.5, ILP4 gaps (2,5)->(2,10), smoothing .8/5->.825/4.5, gap fill vres 2.5->3.
    # 2-model ensemble ddx +.0025 +.0018 +.0030 +.0043 +.0024 (edge part +.0025 +.0018 +.0018 +.0010 +.0024; the rest is
    # 2 fewer false divisions on f2 and f3 from the division weight)
    ilp2 = {"edge_threshold": 0.35, "idsim": 0.5, "appearance_weight": 0.5}
    ilp3 = {**InferConfig.ilp3, "weak_lo": 0.2, "keep_lo": 0.25, "gamma": 0.5, "scaffold_bonus": 0.0,
            "keep_bonus": 0.0, "mid_penalty": 0.0, "idsim": 0.7, "disappearance": 3.5, "edge_weight": -1.0}
    ilp_division_weight = 0.85
    smooth_weight = 0.825
    smooth_max_shift_um = 4.5
    gap_fill = {**InferConfig.gap_fill, "velocity_residual_um": 3.0}
    # K1 = C1 + these three stages (all False: C1). Ensemble OOF proxy vs C1: +.0020 official,
    # every fold >= +.001
    rematch = {**InferConfig.rematch, "enable": False}
    gap_close = {**InferConfig.gap_close, "enable": True, "gaps": (2, 10)}
    adjacent_join = {**InferConfig.adjacent_join, "enable": False}
    # ILP1 (with both prunings), rematch and adjacent_join off: 2-model ensemble ddx vs all on
    # -.0010 -.0021 -.0014 +.0002 +.0004
    ilp1 = {**InferConfig.ilp1, "enable": False}
    temporal_detection_fusion = False
    use_tta = True
    tta_mode = "8fixed"
    tta_head_fusion = True
    tta_idsim_fusion = True
    centroid = {**InferConfig.centroid, "enable": False}  # off: 2-model ensemble ddx f0-f2 +.0007 +.0010 -.0016, ~10% less GPU inference
   #保留 centroid，改成"每帧只读一次原始数据、只算一次？？

class TrainConfig(BaseConfig):
    ckpt_dir = os.path.join(BaseConfig.output_dir, "weights")

    # "zebrahub" trains on external crops while retaining a competition val fold.
    dataset = "competition"  # "competition" or "zebrahub"
    zebrahub_dir = os.path.join(BaseConfig.data_dir, "zebrahub")
    zebrahub_ids = ["ZSNS001", "ZSNS003", "ZSNS004", "ZSNS005"]
    zebrahub_val_fraction = 0.1
    zebrahub_train_samples = 8000
    zebrahub_val_samples = 512

    use_human_division_train_data = True
    use_human_division_val_data = True
    human_division_manifest = os.path.join(
        BaseConfig.data_dir, "manual_review", "human_positive.csv"
    )

    # Initialize from run77 external pretraining; start a new competition run.
    ckpt_path = os.path.join(
        BaseConfig.project_root,
        "outputs/run77/weights/fold_0/epoch_040.pth",
    )
    # "v1": run53 linker; "v2": self + cross attention on multi-scale features.
    link_head = "v2"

    fold_mode = "stratified"
    n_folds = 5
    fold = 0

    use_amp = True
    epochs = 50
    lr = 1e-4
    min_lr = 1e-6
    scheduler = "cosine"  # None or "cosine"
    batch_size = 8
    num_workers = 8
    max_iters = None

    det_loss_weight = 1.0
    det_neg_weight = 0.01
    det_sigma_vox = 1.0
    id_loss_weight = 0.005
    id_tau = 0.1
    appearance_loss_weight = 0.002
    appearance_pos_weight = 10.0
    division_loss_weight = 0.01
    division_pos_weight = 20.0

    aug = "aug01"
    comp_eval_every = 5


# Training frames excluded from CV scoring (key: clip, value: frame t of the pair t -> t+1).
# The hidden test set has neither, so CV removes every edge leaving these frames from prediction
# and truth.
# DUPLICATE_FRAMES: raw volume t is byte-identical to t+1; edges leaving t and t+1 are excluded.
DUPLICATE_FRAMES = {
    "6bba_05b6850b": [4, 12, 27, 42, 52, 57, 59, 62, 66, 76],
    "6bba_05db0fb1": [26, 34, 49, 64, 74, 79, 81, 84, 88, 98],
    "6bba_062c8d37": [7, 12, 36, 37, 38, 42, 50, 54, 59, 66],
    "6bba_07477033": [4, 12, 27, 42, 52, 57, 59, 62, 66, 76],
    "6bba_07e24132": [16, 24, 39, 54, 64, 69, 71, 74, 78, 88],
    "6bba_09961292": [26, 34, 49, 64, 74, 79, 81, 84, 88, 98],
    "6bba_0c7fa718": [16, 37, 46, 48, 55, 68, 73, 97, 98],
    "6bba_0e7c0d07": [27, 66, 87, 96, 98],
    "6bba_12665c0e": [29, 68, 89, 98],
    "6bba_13f19531": [45, 84],
    "6bba_15403b9a": [21, 60, 81, 90, 92],
    "6bba_1d0d8384": [18, 26, 50, 59, 60, 61, 71, 84, 89, 90],
    "6bba_1ebfb80d": [16, 24, 48, 57, 58, 59, 69, 82, 87, 88],
    "6bba_1f58c2f6": [1, 14, 19, 43, 44, 45, 49, 57, 61, 66, 73],
    "6bba_207c6aaf": [48, 87],
    "6bba_20852818": [1, 14, 19, 43, 44, 45, 49, 57, 61, 66, 73],
    "6bba_23af9eeb": [34, 42, 66, 75, 76, 77, 87],
    "6bba_2540cd90": [8, 13, 15, 18, 22, 32, 69, 77],
    "6bba_2646afc7": [17, 25, 49, 58, 59, 60, 70, 83, 88, 89],
    "6bba_268e1230": [27, 48, 57, 59, 66, 79, 84],
    "6bba_2819ca14": [20, 59, 80, 89, 91, 98],
    "6bba_312f0dc3": [79],
    "6bba_32db13fc": [3, 27, 28, 29, 33, 41, 45, 50, 57, 94],
    "6bba_337b1b3a": [76],
    "6bba_372c8cb8": [10, 18, 33, 48, 58, 63, 65, 68, 72, 82],
    "6bba_3a1849c2": [1, 16, 26, 31, 33, 36, 40, 50, 87, 95],
    "6bba_3abfe10a": [5, 18, 23, 47, 48, 49, 53, 61, 65, 70, 77],
    "6bba_3c5691b6": [29, 37, 52, 67, 77, 82, 84, 87, 91],
    "6bba_3db54e20": [35, 56, 65, 67, 74, 87, 92],
    "6bba_3fda6b25": [3, 11, 15, 20, 27, 64, 72, 87],
    "6bba_43fea39d": [3, 16, 21, 45, 46, 47, 51, 59, 63, 68, 75],
    "6bba_474be664": [3, 40, 48, 72, 81, 82, 83, 93],
    "6bba_4aa14c89": [3, 5, 12, 25, 30, 54, 55, 56, 60, 68, 72, 77, 84],
    "6bba_4f99ce20": [7, 9, 16, 29, 34, 58, 59, 60, 64, 72, 76, 81, 88],
    "6bba_55b7eebe": [4, 41, 49, 73, 82, 83, 84, 94],
    "6bba_55c70843": [16, 24, 48, 57, 58, 59, 69, 82, 87, 88],
    "6bba_57b7cc1e": [34, 42, 57, 72, 82, 87, 89, 92, 96],
    "6bba_5a4d9360": [18, 39, 48, 50, 57, 70, 75],
    "6bba_5b28472a": [4, 12, 27, 42, 52, 57, 59, 62, 66, 76],
    "6bba_5f89039d": [3, 8, 15, 52, 60, 75, 90],
    "6bba_61dd1e0d": [7, 11, 16, 23, 60, 68, 83, 98],
    "6bba_61ecbe65": [22, 43, 52, 54, 61, 74, 79],
    "6bba_6321a359": [0, 9, 11, 18, 31, 36, 60, 61, 62, 66, 74, 78, 83, 90],
    "6bba_6479435d": [3, 5, 12, 25, 30, 54, 55, 56, 60, 68, 72, 77, 84],
    "6bba_67ebd073": [9, 14, 38, 39, 40, 44, 52, 56, 61, 68],
    "6bba_6ca87370": [26, 34, 58, 67, 68, 69, 79, 92, 97, 98],
    "6bba_6db0e9b4": [8, 9, 10, 14, 22, 26, 31, 38, 75, 83, 98],
    "6bba_6feb10f0": [6, 43, 51, 75, 84, 85, 86, 96],
    "6bba_6feeb0b1": [33, 41, 65, 74, 75, 76, 86],
    "6bba_705ec2c9": [23, 24, 25, 29, 37, 41, 46, 53, 90, 98],
    "6bba_718b21f9": [3, 13, 18, 20, 23, 27, 37, 74, 82],
    "6bba_74686d6a": [13, 23, 28, 30, 33, 37, 47, 84, 92],
    "6bba_767a1e17": [8, 47, 68, 77, 79, 86],
    "6bba_76db78c1": [4, 11, 48, 56, 71, 86, 96],
    "6bba_786893ac": [14, 22, 37, 52, 62, 67, 69, 72, 76, 86],
    "6bba_789f8168": [29, 50, 59, 61, 68, 81, 86],
    "6bba_78a7bd97": [7, 16, 18, 25, 38, 43, 67, 68, 69, 73, 81, 85, 90, 97],
    "6bba_7af54fde": [11, 16, 40, 41, 42, 46, 54, 58, 63, 70],
    "6bba_7b5d3b2c": [7, 44, 52, 76, 85, 86, 87, 97],
    "6bba_7d3058ae": [3, 11, 26, 41, 51, 56, 58, 61, 65, 75],
    "6bba_7f87b3d8": [0, 21, 30, 32, 39, 52, 57, 81, 82, 83, 87, 95],
    "6bba_80d12824": [1, 14, 19, 43, 44, 45, 49, 57, 61, 66, 73],
    "6bba_825bd1c6": [23, 31, 55, 64, 65, 66, 76, 89, 94, 95],
    "6bba_87289e13": [19, 40, 49, 51, 58, 71, 76],
    "6bba_8b7818bf": [4, 6, 9, 13, 23, 60, 68, 92],
    "6bba_91951b3a": [63],
    "6bba_96833384": [25, 64, 85, 94, 96],
    "6bba_969618f6": [81],
    "6bba_971fa5e0": [0, 7, 20, 25, 49, 50, 51, 55, 63, 67, 72, 79],
    "6bba_9a41d029": [28, 36, 51, 66, 76, 81, 83, 86, 90],
    "6bba_9e23430b": [6, 7, 8, 12, 20, 24, 29, 36, 73, 81, 96],
    "6bba_a5e926bb": [6, 45, 66, 75, 77, 84, 97],
    "6bba_a90a0b9c": [20, 28, 52, 61, 62, 63, 73, 86, 91, 92],
    "6bba_ab78413d": [19, 27, 51, 60, 61, 62, 72, 85, 90, 91],
    "6bba_acd782a8": [48, 87],
    "6bba_ae82a791": [25, 33, 48, 63, 73, 78, 80, 83, 87, 97],
    "6bba_aeee7805": [3, 16, 21, 45, 46, 47, 51, 59, 63, 68, 75],
    "6bba_af149c94": [3, 18, 33, 43, 48, 50, 53, 57, 67],
    "6bba_afb141ff": [23, 24, 25, 29, 37, 41, 46, 53, 90, 98],
    "6bba_b1ae37b9": [0, 15, 25, 30, 32, 35, 39, 49, 86, 94],
    "6bba_b204cac7": [13, 28, 38, 43, 45, 48, 52, 62],
    "6bba_b329af44": [1, 16, 26, 31, 33, 36, 40, 50, 87, 95],
    "6bba_b693381b": [3, 12, 14, 21, 34, 39, 63, 64, 65, 69, 77, 81, 86, 93],
    "6bba_bb9f20c3": [0, 1, 5, 13, 17, 22, 29, 66, 74, 89],
    "6bba_bbb708ca": [8, 10, 17, 30, 35, 59, 60, 61, 65, 73, 77, 82, 89],
    "6bba_c27cba08": [44, 83],
    "6bba_c328f2fd": [0, 2, 9, 22, 27, 51, 52, 53, 57, 65, 69, 74, 81],
    "6bba_c73a1d11": [17, 56, 77, 86, 88, 95],
    "6bba_cdcfe533": [80],
    "6bba_cff5865f": [44, 83],
    "6bba_d0fc38b5": [31, 39, 63, 72, 73, 74, 84, 97],
    "6bba_d1acb6ff": [47, 86],
    "6bba_d3da753b": [32, 71, 92],
    "6bba_d5eae175": [0, 4, 12, 16, 21, 28, 65, 73, 88],
    "6bba_d6ecebbb": [35, 43, 67, 76, 77, 78, 88],
    "6bba_d82a4fc6": [10, 20, 25, 27, 30, 34, 44, 81, 89],
    "6bba_debd7bfa": [5, 44, 65, 74, 76, 83, 96],
    "6bba_e16ffc58": [34, 42, 66, 75, 76, 77, 87],
    "6bba_e5e44988": [13, 52, 73, 82, 84, 91],
    "6bba_ebdf3b34": [6, 7, 8, 12, 20, 24, 29, 36, 73, 81, 96],
    "6bba_ebff6e76": [4, 41, 49, 73, 82, 83, 84, 94],
    "6bba_ed9377fd": [2, 12, 49, 57, 81, 90, 91, 92],
    "6bba_edf14583": [35, 43, 67, 76, 77, 78, 88],
    "6bba_eebc57a5": [9, 46, 54, 78, 87, 88, 89],
    "6bba_ef7b4f7e": [3, 11, 15, 20, 27, 64, 72, 87],
    "6bba_f17befbc": [3, 8, 32, 33, 34, 38, 46, 50, 55, 62],
    "6bba_f1fde7e0": [2, 23, 32, 34, 41, 54, 59, 83, 84, 85, 89, 97],
    "6bba_f20478e9": [1, 10, 12, 19, 32, 37, 61, 62, 63, 67, 75, 79, 84, 91],
    "6bba_f4ae811c": [63],
    "6bba_f8ffd5e7": [81],
    "6bba_fbc898dc": [3, 8, 15, 52, 60, 75, 90],
    "6bba_fc516dc6": [7, 16, 18, 25, 38, 43, 67, 68, 69, 73, 81, 85, 90, 97],
    "6bba_fc83837d": [20, 28, 52, 61, 62, 63, 73, 86, 91, 92],
    "6bba_fe670320": [1, 40, 61, 70, 72, 79, 92, 97],
}
# DRIFT_FRAMES: GT whole-frame drift t -> t+1 (norm of the median GT displacement) > 6 um; edges
# leaving t are excluded. The hidden test set has no frame pair above 6 um (LB probes 2026-09-15).
DRIFT_FRAMES = {
    "44b6_0113de3b": [45, 46, 47, 50],
    "44b6_0db75fae": [16],
    "44b6_1574802b": [52],
    "44b6_267148e4": [10, 13, 29, 30, 44, 45, 46, 48, 51],
    "44b6_2f31fc2f": [26, 27, 28, 30, 31, 56],
    "44b6_341df25f": [14],
    "44b6_3a861e03": [37, 38, 41],
    "44b6_587a1e22": [21, 55],
    "44b6_5f15d135": [0, 96],
    "44b6_668e0cc7": [19, 20, 23],
    "44b6_74d0c52e": [58],
    "44b6_87bba6c4": [17, 20, 37, 46, 50, 51, 52, 53, 54, 55],
    "44b6_95029e92": [19],
    "44b6_97725144": [0],
    "44b6_a21120c2": [10, 41],
    "44b6_c15fded2": [21, 22, 23, 24, 26],
    "44b6_c8e2a523": [1, 2, 5],
    "44b6_cf8fed6b": [25, 56, 57, 60],
    "44b6_d29c9ab2": [49, 50, 53],
    "44b6_d754aa59": [36, 37, 43, 50, 51, 52, 53, 55],
    "44b6_ddf577ad": [55],
    "44b6_e29f0176": [36, 37, 39, 40, 68],
    "44b6_e31261b4": [78],
    "44b6_e57ff5c6": [0, 2, 8, 9, 10, 12, 14, 18, 19, 28, 32, 33, 34, 37, 90, 91],
    "44b6_f28707c6": [22, 42, 51, 52, 54, 55, 56, 57, 58, 59, 60],
    "6bba_05db0fb1": [38, 92, 93],
    "6bba_062c8d37": [2, 8, 26, 32],
    "6bba_07e24132": [75],
    "6bba_09961292": [38, 92, 93],
    "6bba_0c7fa718": [17, 18, 19, 20, 24, 26, 36, 38, 40, 45, 47, 49, 51, 56, 63, 87, 93],
    "6bba_0e7c0d07": [2, 38, 51, 67, 68, 73, 75],
    "6bba_12665c0e": [6, 24, 32, 70, 71, 76, 88],
    "6bba_13f19531": [56, 85],
    "6bba_15403b9a": [12, 16, 22, 61, 68, 79, 93],
    "6bba_1d0d8384": [19, 62],
    "6bba_1ebfb80d": [17, 60, 70],
    "6bba_1f58c2f6": [2, 23, 50, 67],
    "6bba_207c6aaf": [89, 95, 96],
    "6bba_20852818": [2, 50],
    "6bba_2312ac41": [94],
    "6bba_23af9eeb": [35],
    "6bba_2540cd90": [20],
    "6bba_2646afc7": [61, 90],
    "6bba_268e1230": [28, 29, 35, 47, 60, 85],
    "6bba_2819ca14": [21, 60, 78, 81, 92],
    "6bba_312f0dc3": [54, 59, 61, 70],
    "6bba_32db13fc": [17, 23, 59, 65, 93],
    "6bba_337b1b3a": [79],
    "6bba_3abfe10a": [24, 34, 46, 50],
    "6bba_3db54e20": [7, 20, 42, 44, 68, 78],
    "6bba_3fda6b25": [76],
    "6bba_43fea39d": [22, 32, 48],
    "6bba_48816121": [95],
    "6bba_4f99ce20": [1, 10, 12, 17, 24, 37, 48, 54],
    "6bba_55c70843": [17],
    "6bba_5a4d9360": [19, 26, 37, 40, 47, 49, 51, 76, 86, 98],
    "6bba_5b28472a": [8, 9, 64],
    "6bba_5c824876": [96],
    "6bba_5f89039d": [57],
    "6bba_61dd1e0d": [65],
    "6bba_61ecbe65": [23, 24, 41, 42, 44, 53, 55, 80, 90],
    "6bba_6321a359": [1, 3, 12, 19, 26],
    "6bba_6479435d": [33, 49, 51, 61, 67],
    "6bba_67ebd073": [15, 28, 31, 33, 34, 37, 43, 45, 48],
    "6bba_6ca87370": [27, 70, 80],
    "6bba_6db0e9b4": [13],
    "6bba_6feeb0b1": [34],
    "6bba_718b21f9": [25],
    "6bba_74686d6a": [35],
    "6bba_767a1e17": [48, 65, 66, 69, 78, 80],
    "6bba_786893ac": [18, 19, 73],
    "6bba_789f8168": [31, 39, 62, 64, 69, 76],
    "6bba_78a7bd97": [6, 10, 11, 15, 17, 29, 44, 54, 66, 70],
    "6bba_7af54fde": [43],
    "6bba_7d3058ae": [15, 69, 70],
    "6bba_7f87b3d8": [1, 2, 3, 4, 8, 22, 24, 33, 35, 40, 47, 61, 71, 77, 88],
    "6bba_80d12824": [2, 42, 46, 48, 74],
    "6bba_825bd1c6": [67],
    "6bba_87289e13": [4, 20, 27, 28, 38, 41, 50, 52, 87],
    "6bba_8b7818bf": [37, 61],
    "6bba_91951b3a": [64, 74],
    "6bba_96833384": [20, 74, 84],
    "6bba_971fa5e0": [1, 26, 36, 48, 52],
    "6bba_9a41d029": [41],
    "6bba_9e23430b": [5, 9, 65],
    "6bba_a5e926bb": [1, 7, 9, 47, 48, 53, 55, 65, 78, 80, 85, 92],
    "6bba_acd782a8": [39, 43, 49, 59, 88, 95, 96],
    "6bba_ae82a791": [37, 92],
    "6bba_afb141ff": [10, 22, 26],
    "6bba_b204cac7": [50],
    "6bba_b329af44": [44, 88],
    "6bba_b693381b": [4, 6, 15, 17, 22, 29, 53, 59],
    "6bba_bb9f20c3": [78],
    "6bba_bbb708ca": [0, 25, 49, 55, 62],
    "6bba_c328f2fd": [13, 23, 28],
    "6bba_c73a1d11": [12, 20, 58, 66, 76, 78, 80, 89, 91],
    "6bba_cdcfe533": [75, 92],
    "6bba_cff5865f": [47],
    "6bba_d0fc38b5": [32],
    "6bba_d1acb6ff": [34, 42, 48, 82, 88, 89, 94],
    "6bba_d3da753b": [12, 23, 27, 33, 43, 72, 79, 80],
    "6bba_d5eae175": [70],
    "6bba_d82a4fc6": [38],
    "6bba_debd7bfa": [30, 46, 47, 52, 54, 64, 66, 68, 77, 79, 91],
    "6bba_e16ffc58": [35],
    "6bba_e5e44988": [14, 24, 53, 60, 71, 72, 74, 83, 85, 95],
    "6bba_ebdf3b34": [2, 30, 85],
    "6bba_eebc57a5": [3, 4, 47, 90],
    "6bba_f17befbc": [4, 22, 28, 29, 39],
    "6bba_f1fde7e0": [4, 5, 6, 9, 10, 11, 19, 22, 24, 26, 31, 35, 37, 42, 49, 55, 63, 73, 79],
    "6bba_f20478e9": [0, 2, 4, 13, 15, 27, 41, 51, 57, 64, 65, 68],
    "6bba_f4ae811c": [64, 66],
    "6bba_f8ffd5e7": [63, 68, 92],
    "6bba_fc516dc6": [5, 8, 17, 19, 44, 54, 66, 70],
    "6bba_fc83837d": [21, 64],
    "6bba_fe670320": [2, 12, 42, 49, 59, 62, 73, 83, 98],
}
