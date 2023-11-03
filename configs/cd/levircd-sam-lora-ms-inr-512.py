import os

work_dir = './work_dirs/levircd/sam-lora-ms-inr-512'

crop_size = (512, 512)

data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=True,  # convert image from BGR to RGB
)

# 去mmpretrain.ViTSAM的代码仓库下载这个权重
sam_pretrain_load_from = 'pretrain/vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth'

model = dict(
    type='CDEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ViTSAMVisualBackbone',
        model_cfg=dict(
            type='mmpretrain.ViTSAM',
            arch='base',  # 300M params # 3000M params
            img_size=crop_size[0],
            patch_size=16,
            out_channels=256,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=14,
            init_cfg=dict(type='Pretrained', checkpoint=sam_pretrain_load_from, prefix='backbone.')
        ),
        peft_config=dict(
            r=32,
            target_modules=["qkv", "proj"],
            lora_dropout=0.02,
            bias='lora_only',
        )
    ),
    neck=dict(
        type='SimpleFPN',
        backbone_channel=256,
        in_channels=[64, 128, 256, 256],
        out_channels=256,
        num_outs=5,
        norm_cfg=dict(type='LN2d', requires_grad=True)
    ),
    decode_head=dict(
        type='INRSegHead',
        fcn_only=False,
        in_channels=256,
        num_feat_layers=5,
        out_size=(256, 256),
        local_unfold=True,
        sample_mode='nearest',
        align_corners=False,
        coord_map_layers=2,
        num_classes=2,
        loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# dataset settings
dataset_type = 'LEVIRCDDataset'
data_root = '/mnt/search01/dataset/cky_data/levircd512'
data_root = '/mnt/nlp-ali/usr/chenkeyan/dataset/levircd512'


train_pipeline = [
    dict(type='LoadMultipleImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', imdecode_backend='cv2'),
    dict(type='ConcatCDInput'),
    dict(type='Resize', scale=crop_size),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadMultipleImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', imdecode_backend='cv2'),
    dict(type='ConcatCDInput'),
    dict(type='Resize', scale=crop_size),
    dict(type='PackSegInputs')
]

batch_size = 8
num_workers = 8
persistent_workers = True

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train/A', img_path2='train/B', seg_map_path='train/label'),
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

val_evaluator = dict(type='CDMetric')

test_dataloader = val_dataloader
test_evaluator = val_evaluator


default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend', init_kwargs=dict(project='levir-cd', name=os.path.basename(work_dir)))
                ]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

max_epochs = 300
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
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, max_keep_ckpts=5, save_best='cd_metric/jaccard_index_changed', rule='greater', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))