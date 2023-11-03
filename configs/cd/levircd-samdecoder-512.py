import os
# model settings
backbone_norm_cfg = dict(type='LN', requires_grad=True)

crop_size = (512, 512)
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    # to_rgb=True,  # convert image from BGR to RGB
)
num_classes = 2

model = dict(
    type='CDEncoderDecoder',
    data_preprocessor=data_preprocessor,
    init_cfg=dict(type='Pretrained', checkpoint='pretrain/vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth'),
    backbone=dict(
        type='mmpretrain.ViTSAM',
        arch='base',
        img_size=crop_size[0],
        patch_size=16,
        out_channels=256,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
    ),
    neck=dict(
        type='SAMCDMaskDecoder',
        transformer_dim=256,
    ),
    decode_head=dict(
        type='CDPseudoHead',
        num_classes=1,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

find_unused_parameters = True
# dataset settings
dataset_type = 'LEVIRCDDataset'
data_root = '/mnt/search01/dataset/cky_data/levircd512'


train_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(type='LoadAnnotations', imdecode_backend='cv2'),
    dict(type='ConcatCDInput'),
    dict(type='Resize', scale=crop_size),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(type='LoadAnnotations', imdecode_backend='cv2'),
    dict(type='ConcatCDInput'),
    dict(type='Resize', scale=crop_size),
    dict(type='PackSegInputs')
]

batch_size = 8
num_workers = 4
persistent_workers = True

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/A',
            img_path2='train/B',
            seg_map_path='train/label'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/A', img_path2='test/B', seg_map_path='test/label'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])

test_dataloader = val_dataloader
test_evaluator = val_evaluator


default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

work_dir = './work_dirs/levircd/levircd-samdecoder-512-cross'

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend', init_kwargs=dict(project='levircd', name=os.path.basename(work_dir)))
                ]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=True)
log_level = 'INFO'
load_from = None
resume = True

max_epochs = 200
# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1,
        end=max_epochs,
        by_epoch=True,
    )
]

# training schedule for 20k
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, max_keep_ckpts=5, save_best='mIoU', rule='greater', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))