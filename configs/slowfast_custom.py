# SlowFast Configuration for 6-Class Custom Action Recognition
# Based on official MMAction2 SlowFast R50 config

from mmengine.optim import CosineAnnealingLR, LinearLR
from mmaction.models import (ActionDataPreprocessor, Recognizer3D,
                             ResNet3dSlowFast, SlowFastHead)

# Model settings
model = dict(
    type=Recognizer3D,
    backbone=dict(
        type=ResNet3dSlowFast,
        pretrained=None,  # Will load from checkpoint
        resample_rate=8,  # tau
        speed_ratio=8,    # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type=SlowFastHead,
        in_channels=2304,  # 2048+256
        num_classes=6,  # Your 6 custom actions
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'),
    data_preprocessor=dict(
        type=ActionDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

# Dataset type
dataset_type = 'VideoDataset'

# Training pipeline
train_pipeline = [
    dict(type='DecordInit', io_backend='disk'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
    ),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs'),
]

# Test pipeline (for validation)
test_pipeline = [
    dict(type='DecordInit', io_backend='disk'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True,
    ),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs'),
]

# Data loaders
train_dataloader = dict(
    batch_size=2,  # Reduced for 3.8 GiB GPU
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='data/custom_actions_videos_clean/train_list.txt',
        data_prefix=dict(video='data/custom_actions_videos_clean'),
        pipeline=train_pipeline,
        multi_class=False,
        num_classes=6,
    )
)

val_dataloader = dict(
    batch_size=2,  # Reduced for 3.8 GiB GPU
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='data/custom_actions_videos_clean/val_list.txt',
        data_prefix=dict(video='data/custom_actions_videos_clean'),
        pipeline=train_pipeline,  # Use train_pipeline for validation
        test_mode=True,
        multi_class=False,
        num_classes=6,
    )
)

test_dataloader = val_dataloader

# Optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
    ),
    clip_grad=dict(max_norm=40, norm_type=2),
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        T_max=40,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=40)
]

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=40,
    val_begin=1,
    val_interval=1,
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Evaluation metrics
val_evaluator = dict(type='AccMetric')
test_evaluator = dict(type='AccMetric')

# Hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='auto',
    ),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    timer=dict(type='IterTimerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
)

# Runtime settings
work_dir = 'work_dirs/slowfast_custom'
log_level = 'INFO'

# Load pretrained weights for fine-tuning
load_from = 'checkpoints/slowfast_r50_kinetics400_rgb.pth'

# Random seed
randomness = dict(seed=42, deterministic=False)

# Default scope
default_scope = 'mmaction'

# Environment
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Visualizer
vis_backends = [dict(type='LocalVisBackend')]
