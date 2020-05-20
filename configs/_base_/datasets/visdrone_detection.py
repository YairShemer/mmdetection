dataset_type = 'VisdroneDataset'
data_root = 'data/visdrone/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 500), keep_ratio=True), # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 500), # img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,#2
    workers_per_gpu=0, #2
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'VisDrone2019-DET-train',
        img_prefix=data_root + 'VisDrone2019-DET-train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VisDrone2019-DET-val',
        img_prefix=data_root + 'VisDrone2019-DET-val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VisDrone2019-DET-test-dev',
        img_prefix=data_root + 'VisDrone2019-DET-test-dev',
        # ann_file=data_root + 'VisDrone2019-DET-train',
        # img_prefix=data_root + 'VisDrone2019-DET-train',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='recall')
