import os


class BaseConfig:
    run = "run53"

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
    unet_layers = (32, 64, 128)
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
        # "run53/fold_1/epoch_040.pth",
    ]
    use_amp = True

    point_threshold = 0.90
    temporal_detection_fusion = False
    use_tta = False
    tta_mode = "8fixed"
    tta_head_fusion = False
    tta_idsim_fusion = False
    parabolic_refinement = True
    edge_threshold = 0.35

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
        "enable": False,
        "weight": 0.4,
    }
    centroid = {
        "enable": False,
        "blend": 0.5,
        "rz": 2,
        "ryx": 5,
    }
    gap_fill = {
        "enable": False,
        "gate_um": 10.0,
        "velocity_residual_um": 2.5,
        "occupied_um": 3.0,
    }
    # Penalise candidate edges whose displacement deviates from the collective
    # motion of neighbouring detections. In whole-frame drift the link head
    # swaps every cell with the neighbour that moved into its old position;
    # the neighbourhood flow is the cue it lacks. 3-fold CV adjJ +0.0045.
    flow_cost = {
        "enable": False,
        "weight": 0.2,
        "knee_um": 2.0,          # residual below this is free
        "neighbours": 15,        # matched pairs per node for the regional flow
        "drift_search_um": 12.0, # whole-frame drift grid search range
    }
    track_postprocess = {
        "mode": "none",  # "none", "no_new_divisions" or "anchor_paths"
        "low_point_threshold": 0.6,
    }

    smooth_window = 4
    smooth_weight = 0.8
    smooth_preserve_motion = True  # Preserve common frame motion while smoothing.

    max_distance = 7.0
    adjustment_alpha = 0.1
    division_weight = 0.1


class SubmitConfig(InferConfig):
    track_postprocess = {**InferConfig.track_postprocess, "mode": "no_new_divisions"}
    temporal_detection_fusion = False
    use_tta = True
    tta_mode = "8fixed"
    tta_head_fusion = True
    tta_idsim_fusion = True
    idsim = {**InferConfig.idsim, "enable": True}
    centroid = {**InferConfig.centroid, "enable": True}
    gap_fill = {**InferConfig.gap_fill, "enable": True}


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

    ckpt_path = os.path.join(
        BaseConfig.project_root,
        "outputs/run19/weights/fold_0/epoch_075.pth",
    )

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
    det_neg_weight = 0.01 # 1 # 0.01 ！！！！！
    det_sigma_vox = 1.0
    id_loss_weight = 0.005
    id_tau = 0.1
    appearance_loss_weight = 0.002
    appearance_pos_weight = 10.0
    division_loss_weight = 0.01
    division_pos_weight = 20.0

    aug = "aug01"
    comp_eval_every = 5
