# Model configuration for 6-class action recognition
model = dict(
    type="RecognizerGCN",
    backbone=dict(
        type="STGCN",
        graph_cfg=dict(layout="coco", mode="stgcn_spatial"),
        in_channels=3,  # 3D keypoints (x, y, score)
    ),
    cls_head=dict(
        type="GCNHead",
        num_classes=6,  # Your 6 custom actions
        in_channels=256,
    ),
)

# Dataset configuration
dataset_type = "PoseDataset"

# Training pipeline
train_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="PackActionInputs"),
]

# Validation pipeline
val_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="PackActionInputs"),
]

# Test pipeline
test_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
    dict(type="UniformSampleFrames", clip_len=100, num_clips=10, test_mode=True),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="PackActionInputs"),
]

# Data loader configuration
train_dataloader = dict(
    batch_size=8,  # Adjust based on GPU memory
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file="data/custom_actions_processed/train_data.pkl",
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file="data/custom_actions_processed/val_data.pkl",
        pipeline=val_pipeline,
        test_mode=True,
    ),
)

test_dataloader = val_dataloader

# Optimizer configuration
optim_wrapper = dict(
    optimizer=dict(
        type="SGD",
        lr=0.01,  # Lower learning rate for fine-tuning
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True,
    ),
    paramwise_cfg=dict(
        # Fine-tuning: slower learning for backbone
        custom_keys={
            "backbone": dict(lr_mult=0.1),
        }
    ),
    clip_grad=dict(max_norm=10, norm_type=2),
)

# Learning rate scheduler
param_scheduler = [
    dict(type="MultiStepLR", by_epoch=True, milestones=[5, 10, 15], gamma=0.1)
]

# Training configuration
train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=20,  # Adjust based on dataset size
    val_begin=1,
    val_interval=1,
)

val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# Evaluation
val_evaluator = [dict(type="AccMetric")]
test_evaluator = val_evaluator

# Hooks
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook", interval=1, max_keep_ckpts=3, save_best="auto"
    ),
    logger=dict(type="LoggerHook", interval=10),
    param_scheduler=dict(type="ParamSchedulerHook"),
    timer=dict(type="IterTimerHook"),
    runtime_info=dict(type="RuntimeInfoHook"),
)

# Runtime settings
work_dir = "work_dirs/custom_action_recognition"
log_level = "INFO"

# Load from checkpoint for fine-tuning
# load_from = "checkpoints/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-2d_20221129-612416c6.pth"
load_from = None

# Random seed
randomness = dict(seed=42, deterministic=False)

# Log processor
log_processor = dict(by_epoch=False)

# Default runtime settings (from mmaction base config)
default_scope = "mmaction"
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
    launcher="pytorch",
)

vis_backends = [dict(type="LocalVisBackend")]
