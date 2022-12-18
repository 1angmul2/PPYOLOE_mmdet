checkpoint_config = dict(interval=30)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(type='SetEpochInfoHook'),
    dict(type='SyncNormHook', num_last_epochs=15, interval=10, priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0002,
        priority=49)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
img_scale = (640, 640)
width_mult = 0.5
depth_mult = 0.33
neck_head_channels = [768, 384, 192]
model = dict(
    type='PPYOLOE',
    sybn=True,
    input_size=(640, 640),
    size_multiplier=32,
    random_size_range=(10, 25),
    random_size_interval=1,
    backbone=dict(
        type='CSPResNet',
        layers=[3, 6, 6, 3],
        channels=[64, 128, 256, 512, 1024],
        return_idx=[1, 2, 3],
        depth_wise=False,
        use_large_stem=True,
        width_mult=0.5,
        depth_mult=0.33,
        act='swish',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pkgs/_CSPResNetb_s_pretrained.pth')),
    neck=dict(
        type='CustomCSPPAN',
        in_channels=[256, 512, 1024],
        out_channels=[768, 384, 192],
        stage_num=1,
        block_num=3,
        width_mult=0.5,
        depth_mult=0.33,
        act='swish',
        spp=True),
    bbox_head=dict(
        type='PPYOLOEHead',
        num_classes=80,
        in_channels=[768, 384, 192],
        width_mult=0.5,
        depth_mult=0.33,
        strides=[32, 16, 8],
        reg_max=16,
        grid_cell_scale=5.0,
        grid_cell_offset=0.5,
        use_varifocal_loss=True,
        eval_input_size=[640, 640],
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='Swish'),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            weight_grad=True,
            alpha=0.75,
            gamma=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.5),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.5)),
    train_cfg=dict(
        initial_epoch=100,
        static_assigner=dict(type='PPYOLOEATSSAssigner', topk=9),
        assigner=dict(
            type='PPYOLOETaskAlignedAssigner', topk=13, alpha=1.0, beta=6.0)),
    test_cfg=dict(
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300,
        nms_pre=1000))
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_train2017.json',
        img_prefix='data/coco/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=18,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(type='Expand', mean=(123.675, 116.28, 103.53), to_rgb=True),
            dict(type='MinIoURandomCrop'),
            dict(
                type='Resize',
                img_scale=[(640, 640)],
                multiscale_mode='value',
                keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(
                type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        size_divisor=32,
                        pad_val=dict(img=(103.53, 116.28, 123.675))),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(
                type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize',
                        img_scale=[(640, 640)],
                        multiscale_mode='value',
                        keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
max_epochs = 300
num_last_epochs = 15
interval = 10
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,
    num_last_epochs=15,
    min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=300)
evaluation = dict(
    save_best='auto', interval=10, dynamic_intervals=[(290, 1)], metric='bbox')
find_unused_parameters = True
gpu_ids = range(0, 8)
