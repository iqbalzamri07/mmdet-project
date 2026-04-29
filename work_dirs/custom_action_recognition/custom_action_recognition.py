dataset_type = 'PoseDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=3, save_best='auto', type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    launcher='pytorch',
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        graph_cfg=dict(layout='coco', mode='stgcn_spatial'),
        in_channels=3,
        type='STGCN'),
    cls_head=dict(in_channels=256, num_classes=6, type='GCNHead'),
    type='RecognizerGCN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(
        lr=0.01, momentum=0.9, nesterov=True, type='SGD', weight_decay=0.0005),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))))
param_scheduler = [
    dict(
        by_epoch=True, gamma=0.1, milestones=[
            5,
            10,
            15,
        ], type='MultiStepLR'),
]
randomness = dict(deterministic=False, seed=42)
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='data/custom_actions_processed/val_data.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(dataset='coco', feats=[
                'j',
            ], type='GenSkeFeat'),
            dict(type='PoseDecode'),
            dict(num_person=2, type='FormatGCNInput'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='PoseDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(type='AccMetric'),
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(dataset='coco', feats=[
        'j',
    ], type='GenSkeFeat'),
    dict(
        clip_len=100, num_clips=10, test_mode=True,
        type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(num_person=2, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=20, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='data/custom_actions_processed/train_data.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(dataset='coco', feats=[
                'j',
            ], type='GenSkeFeat'),
            dict(type='PoseDecode'),
            dict(num_person=2, type='FormatGCNInput'),
            dict(type='PackActionInputs'),
        ],
        type='PoseDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(dataset='coco', feats=[
        'j',
    ], type='GenSkeFeat'),
    dict(type='PoseDecode'),
    dict(num_person=2, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='data/custom_actions_processed/val_data.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(dataset='coco', feats=[
                'j',
            ], type='GenSkeFeat'),
            dict(type='PoseDecode'),
            dict(num_person=2, type='FormatGCNInput'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='PoseDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='AccMetric'),
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(dataset='coco', feats=[
        'j',
    ], type='GenSkeFeat'),
    dict(type='PoseDecode'),
    dict(num_person=2, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
work_dir = 'work_dirs/custom_action_recognition'
