# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import math
from abc import ABCMeta
from typing import List, Optional, Tuple, Any, Sequence, Dict, OrderedDict

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks.transformer import build_transformer_layer, FFN
from mmengine import mkdir_or_exist, MMLogger
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from mmengine.model import BaseModel, BaseModule
from peft import get_peft_config, get_peft_model
from prettytable import PrettyTable
from torch import Tensor
from torchmetrics.functional.classification import multiclass_precision, multiclass_recall, multiclass_f1_score, \
    multiclass_jaccard_index, multiclass_accuracy, binary_accuracy

from mmpretrain.models import CrossMultiheadAttention
from mmseg.registry import MODELS, METRICS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix, MultiConfig)
from .encoder_decoder import EncoderDecoder
from .. import FCNHead, accuracy
from ..decode_heads.decode_head import BaseDecodeHead
from ..utils import resize


@MODELS.register_module()
class ViTSAMVisualBackbone(BaseModule):
    def __init__(self,
                 model_cfg,
                 peft_config=None,
                 init_cfg=None,
                 ):
        super(ViTSAMVisualBackbone, self).__init__(init_cfg)
        default_peft_config = {
            "peft_type": "LORA",
            "task_type": "none",
            "inference_mode": False,
            "r": 32,
            # "target_modules": ["attn_qkv", "attn_outproj", "c_fc", "c_proj"],
            "target_modules": ["attn_qkv", "attn_outproj"],
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "fan_in_fan_out": False,
            "bias": "lora_only",
        }
        if peft_config is not None:
            default_peft_config.update(peft_config)
            peft_config = get_peft_config(default_peft_config)

        model = MODELS.build(model_cfg)
        if peft_config is not None:
            self.peft_model = get_peft_model(model, peft_config)
            if is_main_process():
                self.peft_model.print_trainable_parameters()
        else:
            self.peft_model = model

    def forward(self, x):
        x = self.peft_model(x)
        return x


@MODELS.register_module()
class SimpleFCNHead(BaseModule):
    def __init__(self,
                 in_channels,
                 num_classes,
                 num_upsamplings=1,
                 num_convs=2,
                 threshold=0.5,
                 align_corners=True,
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 init_cfg=None,
                 ):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.align_corners = align_corners
        self.out_channels = num_classes

        up_layers = []
        for i in range(num_upsamplings):
            up_layers.append(
                nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.up_layers = nn.Sequential(*up_layers)

        conv_layers = []
        for i in range(num_convs):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.conv_layers = nn.Sequential(*conv_layers)
        self.seg_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_classes, kernel_size=1)
        )

        self.seg_loss = MODELS.build(loss)

    def forward(self, inputs):
        x = self.up_layers(inputs)
        x = self.conv_layers(x)
        x = self.seg_layer(x)
        return x

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList, train_cfg: ConfigType) -> dict:

        gt_semantic_segs = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
        seg_label = torch.stack(gt_semantic_segs, dim=0)
        seg_label = seg_label.squeeze(1)

        seg_logits = self.forward(inputs)
        seg_logits = resize(input=seg_logits, size=seg_label.shape[-2:], mode='bilinear', align_corners=True)

        losses = dict()

        losses['loss_ce'] = self.seg_loss(seg_logits, seg_label, ignore_index=self.ignore_index)
        losses['acc_seg'] = accuracy(seg_logits, seg_label, ignore_index=self.ignore_index)
        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas, test_cfg: ConfigType) -> OptSampleList:
        seg_logits = self.forward(inputs)
        seg_logits = resize(input=seg_logits, size=batch_img_metas[0]['img_shape'], mode='bilinear', align_corners=True)
        return seg_logits


@MODELS.register_module()
class CDEncoderDecoder(EncoderDecoder):
    def __init__(self,
                 fusion_neck=None,
                 absminus=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.absminus = absminus
        if fusion_neck is not None:
            self.fusion_neck = MODELS.build(fusion_neck)

    @property
    def with_fusion_neck(self) -> bool:
        return hasattr(self, 'fusion_neck') and self.fusion_neck is not None

    def extract_feat(self, inputs):
        # split inputs into two parts
        x0 = inputs[:, :3, :, :]
        x1 = inputs[:, 3:, :, :]
        x = torch.cat([x0, x1], dim=0)
        x = self.backbone(x)
        # only output the last feature map
        x = x[0]
        # restore the two parts
        x0 = x[:x.shape[0] // 2, :, :, :]
        x1 = x[x.shape[0] // 2:, :, :, :]
        if self.with_fusion_neck:
            x0, x1 = self.fusion_neck(x0, x1)
        if self.with_neck:
            x = x0 - x1
            if self.absminus:
                x = torch.abs(x)
            x = self.neck(x)
        else:
            x = x0 - x1
            if self.absminus:
                x = torch.abs(x)
        return x

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        x = self.extract_feat(inputs)
        losses = self.decode_head.loss(x, data_samples, self.train_cfg)
        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]

        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas, self.test_cfg)
        return self.postprocess_result(seg_logits, data_samples)


@METRICS.register_module()
class CDMetric(BaseMetric):
    default_prefix: Optional[str] = 'cd_metric'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
            self.results.append((pred_label, label))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        num_classes = len(self.dataset_meta['classes'])
        class_names = self.dataset_meta['classes']

        assert num_classes == 2, 'Only support binary classification in CDMetric.'

        logger: MMLogger = MMLogger.get_current_instance()
        pred_label, label = zip(*results)
        preds = torch.stack(pred_label, dim=0)
        target = torch.stack(label, dim=0)

        multiclass_precision_ = multiclass_precision(preds, target, num_classes=num_classes, average=None)
        multiclass_recall_ = multiclass_recall(preds, target, num_classes=num_classes, average=None)
        multiclass_f1_score_ = multiclass_f1_score(preds, target, num_classes=num_classes, average=None)
        multiclass_jaccard_index_ = multiclass_jaccard_index(preds, target, num_classes=num_classes, average=None)
        binary_accuracy_ = binary_accuracy(preds, target)
        metrics = dict()
        for i in range(num_classes):
            metrics['precision_' + class_names[i]] = multiclass_precision_[i].item()
            metrics['recall_' + class_names[i]] = multiclass_recall_[i].item()
            metrics['f1_score_' + class_names[i]] = multiclass_f1_score_[i].item()
            metrics['jaccard_index_' + class_names[i]] = multiclass_jaccard_index_[i].item()
        metrics['accuracy'] = binary_accuracy_.item()

        class_table_data = PrettyTable()
        for key, val in metrics.items():
            class_table_data.add_column(key, [val])
        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        return metrics


@MODELS.register_module()
class TransformerDecoder(BaseModule):
    def __init__(self,
                 max_size=256,
                 embed_dim=256,
                 hidden_dim=256 * 4,
                 num_heads=8,
                 num_layers=3,
                 dropout=0.1,
                 activation_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation_cfg = activation_cfg
        self.norm_cfg = norm_cfg
        self.max_size = max_size

        transformer_layer_cfg = dict(
            type='BaseTransformerLayer',
            attn_cfgs=
            dict(
                type='MultiheadAttention',
                embed_dims=embed_dim,
                num_heads=num_heads,
                attn_drop=dropout,
                proj_drop=dropout,
                dropout_layer=dict(type='Dropout', drop_prob=dropout)
            ),
            # attn_cfgs=[
            #     dict(
            #         type='MultiheadAttention',
            #         embed_dims=embed_dim,
            #         num_heads=num_heads,
            #         attn_drop=dropout,
            #         proj_drop=dropout,
            #         dropout_layer=dict(type='Dropout', drop_prob=dropout)
            #     ),
            #     dict(
            #         type='MultiheadAttention',
            #         embed_dims=embed_dim,
            #         num_heads=num_heads,
            #         attn_drop=dropout,
            #         proj_drop=dropout,
            #         dropout_layer=dict(type='Dropout', drop_prob=dropout)
            #     ),
            # ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=embed_dim,
                feedforward_channels=hidden_dim,
                num_fcs=2,
                act_cfg=activation_cfg,
                ffn_drop=dropout,
                add_identity=True),
            operation_order=('norm', 'self_attn', 'norm', 'ffn'),
            norm_cfg=norm_cfg,
            batch_first=True
        )

        self.layers = nn.ModuleList()
        transformer_layers = [
            copy.deepcopy(transformer_layer_cfg) for _ in range(self.num_layers)
        ]
        for layer_cfg in transformer_layers:
            self.layers.append(build_transformer_layer(layer_cfg))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_size, embed_dim))
        # self.filter_bx = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.filter_cd1 = nn.Parameter(torch.ones(1, 2, embed_dim) / embed_dim)
        self.filter_cd2 = nn.Parameter(- torch.ones(1, 2, embed_dim) / embed_dim)
        # self.query_embed = nn.Parameter(torch.zeros(1, self.max_size, embed_dim))
        # self.query_pos_embed = nn.Parameter(torch.zeros(1, self.max_size, embed_dim))
        # nn.init.normal_(self.query_embed, std=0.02)

    def forward(self, x):
        x0 = x[0][0]
        x1 = x[1][0]
        b, c, h, w = x0.shape
        x0 = einops.rearrange(x0, 'b c h w -> b (h w) c')
        x0 = x0 + self.pos_embed
        x1 = einops.rearrange(x1, 'b c h w -> b (h w) c')
        x1 = x1 + self.pos_embed

        filter_cd1 = einops.repeat(self.filter_cd1, 'b n c -> (b bs) n c', bs=b)
        filter_cd2 = einops.repeat(self.filter_cd2, 'b n c -> (b bs) n c', bs=b)

        x = torch.cat([filter_cd1, filter_cd2, x0, x1], dim=1)
        # query = einops.repeat(self.query_embed, 'b n c -> (b bs) n c', bs=b)
        # query = query + self.pos_embed

        # x = torch.cat([query, x], dim=1)

        # key_pos_embed = einops.repeat(self.pos_embed, 'b n c -> b (n nt) c', nt=2)
        for layer in self.layers:
            x = layer(x)
        filter_cd1 = x[:, :2, :]
        filter_cd2 = x[:, 2:4, :]
        x0 = x[:, 4:4 + self.max_size, :]
        x1 = x[:, 4 + self.max_size:, :]

        x0 = einops.rearrange(x0, 'b (h w) c -> b c h w', h=h, w=w)
        x0 = F.interpolate(x0, scale_factor=4, mode='bilinear', align_corners=True)

        x1 = einops.rearrange(x1, 'b (h w) c -> b c h w', h=h, w=w)
        x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=True)

        filter_cd = torch.cat([filter_cd1, filter_cd2], dim=-1)

        return x0, x1, filter_cd


@MODELS.register_module()
class CDFCNHead(FCNHead):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

        kernel_size = self.kernel_size
        dilation = 1
        num_convs = self.num_convs

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs_copy = nn.Identity()
        else:
            self.convs_copy = nn.Sequential(*convs)

        self.conv1 = ConvModule(
            self.channels,
            self.channels,
            kernel_size=kernel_size,
            padding=conv_padding,
            dilation=dilation,
            conv_cfg=self.conv_cfg,
            bias=True,
            act_cfg=None
        )
        self.conv2 = ConvModule(
            self.channels,
            self.channels,
            kernel_size=kernel_size,
            padding=conv_padding,
            dilation=dilation,
            conv_cfg=self.conv_cfg,
            bias=True,
            act_cfg=None
        )

    def _forward_feature(self, x0, x1):
        feats_x0 = self.convs(x0)
        feats_x0 = self.conv1(feats_x0)
        feats_x1 = self.convs_copy(x1)
        feats_x1 = self.conv2(feats_x1)
        feats = torch.cat([feats_x0, feats_x1], dim=1)
        return feats

    def forward(self, inputs):
        """Forward function."""
        x0, x1, filter_cd = inputs
        output = self._forward_feature(x0, x1)
        output = einops.einsum(output, filter_cd, 'b c h w, b n c -> b n h w')
        return output


@MODELS.register_module()
class SAMCDMaskDecoder(nn.Module):
    def __init__(
            self,
            transformer_dim: int,
            activation=nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer(
            depth=3,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        )
        self.pe_emb = PositionEmbeddingRandom(transformer_dim // 2)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            activation(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 2),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2),
            activation(),
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, transformer_dim // 4, iou_head_depth
        )

        self.cd_token = nn.Embedding(1, transformer_dim)

    def forward(self, inputs):
        """Forward function."""
        x0 = inputs[0][0]
        x1 = inputs[1][0]
        b, c, h, w = x0.shape

        out_tokens = einops.repeat(self.cd_token.weight, 'n c -> bs n c', bs=b)

        src = x0 - x1
        pos_src = self.pe_emb((h, w))
        pos_src = einops.repeat(pos_src, 'h w c -> bs h w c', bs=b)

        # Run the transformer
        hs, src = self.transformer(src, pos_src, out_tokens)
        token_out = hs[:, 0:1, :]

        cd_iou_pred = self.iou_prediction_head(token_out)
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        hyper_in = cd_iou_pred
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        return masks


@MODELS.register_module()
class SAMBXMaskDecoder(nn.Module):
    def __init__(
            self,
            transformer_dim: int,
            activation=nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer(
            depth=3,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        )
        self.pe_emb = PositionEmbeddingRandom(transformer_dim // 2)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            activation(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 2),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2),
            activation(),
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, transformer_dim // 4, iou_head_depth
        )

        self.bx_token = nn.Embedding(1, transformer_dim)

    def forward(self, inputs):
        """Forward function."""
        x0 = inputs[0]
        b, c, h, w = x0.shape

        out_tokens = einops.repeat(self.bx_token.weight, 'n c -> bs n c', bs=b)

        src = x0
        pos_src = self.pe_emb((h, w))
        pos_src = einops.repeat(pos_src, 'h w c -> bs h w c', bs=b)

        # Run the transformer
        hs, src = self.transformer(src, pos_src, out_tokens)
        token_out = hs[:, 0:1, :]

        cd_iou_pred = self.iou_prediction_head(token_out)
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        hyper_in = cd_iou_pred
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        return masks


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x



class Sirens(BaseModule):
    def __init__(
            self,
            num_layers,
            input_dims,
            hidden_dims,
            output_dims,
            **kwargs):
        super().__init__(**kwargs)
        self.first_layer = nn.Sequential(
            nn.Conv2d(input_dims, hidden_dims, 1),
            nn.ReLU()
        )

        self.inner_layers = nn.ModuleList()
        for i in range(num_layers):
            self.inner_layers.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims, hidden_dims, 1),
                    nn.ReLU(),
                )
            )
        self.last_layer = nn.Sequential(
            nn.Conv2d(hidden_dims, output_dims, 1),
        )

    def forward(self, x):
        x = self.first_layer(x)
        x = torch.sin(x)
        for layer in self.inner_layers:
            residual = layer(x)
            residual = torch.sin(residual)
            x = x + residual
        x = self.last_layer(x)
        return x

@MODELS.register_module()
class INRSegHead(BaseModule):
    def __init__(self,
                 in_channels=256,
                 fcn_only=False,
                 num_feat_layers=5,
                 out_size=(256, 256),
                 local_unfold=True,
                 sample_mode='bilinear',
                 align_corners=True,
                 coord_map_layers=3,
                 num_classes=2,
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 init_cfg=None,
                 ):
        super().__init__(init_cfg)
        self.fcn_only = fcn_only
        self.in_channels = in_channels
        self.out_size = out_size
        if local_unfold:
            self.local_unfold = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            )
        else:
            self.local_unfold = None

        self.sample_mode = sample_mode
        self.align_corners = align_corners
        self.num_classes = num_classes

        if not fcn_only:
            self.coord_map = Sirens(coord_map_layers, 2+2+2, in_channels, in_channels)
            num_feat_layers = num_feat_layers*2

        self.seg_layer = nn.Sequential(
            nn.Conv2d(in_channels*num_feat_layers, in_channels // 4 // num_feat_layers, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4 // num_feat_layers, num_classes, kernel_size=1)
        )
        self.seg_loss = MODELS.build(loss)
        self.out_channels = num_classes

    def _to_coordinates(self, size=(56, 56), return_map=True):
        """Converts an image to a set of coordinates and features.
        Args:
            img (torch.Tensor): Shape (channels, height, width).
        """
        # H, W
        # Coordinates are indices of all non zero locations of a tensor of ones of
        # same shape as spatial dimensions of image
        # (H*W) 2
        coordinates = torch.nonzero(torch.ones(size), as_tuple=False).float()
        # Normalize coordinates to lie in [-.5, .5]
        coordinates[..., 0] = coordinates[..., 0] / (size[0] - 1) - 0.5
        coordinates[..., 1] = coordinates[..., 1] / (size[1] - 1) - 0.5
        # Convert to range [-1, 1]
        coordinates *= 2
        if return_map:
            coordinates = rearrange(coordinates, '(H W) C -> H W C', H=size[0])
        # [y, x]
        return coordinates

    def get_coordinate_map(self, lr_size, hr_size, device):
        lr_h, lr_w = lr_size
        hr_h, hr_w = hr_size
        # H, W, 2 -> 1, 2, H, W
        lr_coord = self._to_coordinates(lr_size, return_map=True).to(device).permute(2, 0, 1).unsqueeze(0)
        hr_coord = self._to_coordinates(hr_size, return_map=True).to(device).permute(2, 0, 1).unsqueeze(0)
        # important! mode='nearest' gives inconsistent results
        diff_grid = hr_coord - F.interpolate(lr_coord, size=hr_size, mode='nearest-exact')
        diff_grid[:, 0, :, :] *= lr_h
        diff_grid[:, 1, :, :] *= lr_w
        return diff_grid.contiguous(), lr_coord.contiguous(), hr_coord.contiguous()

    def forward(self, inputs):
        if not isinstance(inputs, tuple):
            inputs = (inputs, )
        # inputs is a tuple of multi-level features
        concat_features = []
        device = inputs[0].device
        if self.training:
            if isinstance(self.out_size, dict):
                out_size = torch.randint(self.out_size['min'], self.out_size['max'], (2,)).tolist()
            elif isinstance(self.out_size, tuple):
                out_size = self.out_size
            else:
                raise ValueError('out_size should be a tuple or dict of (min, max)')
        else:
            if isinstance(self.out_size, dict):
                out_size = (self.out_size['max'], self.out_size['max'])
            elif isinstance(self.out_size, tuple):
                out_size = self.out_size
            else:
                raise ValueError('out_size should be a tuple or dict of (min, max)')

        for x in inputs:
            lr_size = x.shape[-2:]

            # Expand funcmap
            if self.local_unfold is not None:
                x = self.local_unfold(x)
            x = F.interpolate(x, size=out_size, mode=self.sample_mode)
            if not self.fcn_only:
                diff_grid, lr_coord, hr_coord = self.get_coordinate_map(lr_size, out_size, device)

                diff_grid = repeat(diff_grid, 'b c h w -> (B b) c h w', B=x.size(0))
                hr_coord = repeat(hr_coord, 'b c h w -> (B b) c h w', B=x.size(0))

                h_ratio = lr_size[0] / out_size[0]
                w_ratio = lr_size[1] / out_size[1]
                scale_ratio_map = torch.tensor([h_ratio, w_ratio]).view(1, -1, 1, 1).expand(x.size(0), -1, *out_size).to(device)

                coord_inputs = torch.cat([diff_grid, scale_ratio_map, hr_coord], dim=1)
                coord_inputs = self.coord_map(coord_inputs)
                x = torch.cat([x, coord_inputs], dim=1)
            concat_features.append(x)
        coord_inputs = torch.cat(concat_features, dim=1)
        x = self.seg_layer(coord_inputs)
        return x

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList, train_cfg: ConfigType) -> dict:
        gt_semantic_segs = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
        seg_label = torch.stack(gt_semantic_segs, dim=0)
        seg_label = seg_label.squeeze(1)

        seg_logits = self.forward(inputs)
        seg_logits = resize(input=seg_logits, size=seg_label.shape[-2:], mode='bilinear', align_corners=True)

        losses = dict()
        losses['loss_ce'] = self.seg_loss(seg_logits, seg_label)
        losses['acc_seg'] = accuracy(seg_logits, seg_label)
        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas, test_cfg: ConfigType) -> OptSampleList:
        seg_logits = self.forward(inputs)
        seg_logits = resize(input=seg_logits, size=batch_img_metas[0]['img_shape'], mode='bilinear', align_corners=True)
        return seg_logits


@MODELS.register_module()
class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@MODELS.register_module()
class SimpleFPN(BaseModule):
    """Simple Feature Pyramid Network for ViTDet."""

    def __init__(self,
                 backbone_channel: int,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2),
            build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],
            nn.GELU(),
            nn.ConvTranspose2d(self.backbone_channel // 2,
                               self.backbone_channel // 4, 2, 2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, input: Tensor) -> tuple:
        """Forward function.

        Args:
            inputs (Tensor): Features from the upstream network, 4D-tensor
        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # build FPN
        inputs = []
        inputs.append(self.fpn1(input))
        inputs.append(self.fpn2(input))
        inputs.append(self.fpn3(input))
        inputs.append(self.fpn4(input))

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)


@MODELS.register_module()
class TransformerFusionNeck(BaseModule):
    def __init__(self, n_layers, num_tokens, embed_dims=256, num_heads=8):
        super().__init__()
        self.n_layers = n_layers
        self.norm_layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.pe_emb = nn.Parameter(torch.zeros(1, num_tokens, embed_dims))

        for i in range(n_layers):
            self.norm_layers.append(
                nn.ModuleList([nn.LayerNorm(embed_dims), nn.LayerNorm(embed_dims)])
            )
            self.cross_attn_layers.append(
                nn.ModuleList([
                    CrossMultiheadAttention(
                        embed_dims,
                        num_heads=num_heads,
                        qkv_bias=True,
                        qk_scale=None,
                        attn_drop=0.1,
                        proj_drop=0.1),
                    CrossMultiheadAttention(
                        embed_dims,
                        num_heads=num_heads,
                        qkv_bias=True,
                        qk_scale=None,
                        attn_drop=0.1,
                        proj_drop=0.1)])
            )
            self.ffn_layers.append(
                nn.ModuleList([
                    FFN(
                        embed_dims=embed_dims,
                        feedforward_channels=embed_dims * 2,
                        num_fcs=2,
                        ffn_drop=0.1,
                        dropout_layer=None,
                        act_cfg=dict(type='GELU'),
                        add_identity=True),
                    FFN(
                        embed_dims=embed_dims,
                        feedforward_channels=embed_dims * 2,
                        num_fcs=2,
                        ffn_drop=0.1,
                        dropout_layer=None,
                        act_cfg=dict(type='GELU'),
                        add_identity=True)]
                )
            )

    def forward(self, x0, x1):
        b, c, h, w = x0.shape
        x0 = einops.rearrange(x0, 'b c h w -> b (h w) c')
        x1 = einops.rearrange(x1, 'b c h w -> b (h w) c')

        for i in range(self.n_layers):
            x0 = x0 + self.pe_emb
            x1 = x1 + self.pe_emb
            x0 = self.norm_layers[i][0](x0)
            x1 = self.norm_layers[i][1](x1)
            x0 = self.cross_attn_layers[i][0](x=x0, k=x1, v=x1)
            x1 = self.cross_attn_layers[i][1](x=x1, k=x0, v=x0)
            x0 = self.ffn_layers[i][0](x0)
            x1 = self.ffn_layers[i][1](x1)
        x0 = einops.rearrange(x0, 'b (h w) c -> b c h w', h=h, w=w)
        x1 = einops.rearrange(x1, 'b (h w) c -> b c h w', h=h, w=w)
        return x0, x1



class TwoWayTransformer(nn.Module):
    def __init__(
            self,
            depth: int,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int,
            activation=nn.ReLU,
            attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
            self,
            image_embedding: Tensor,
            image_pe: Tensor,
            point_embedding: Tensor,
    ):
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int = 2048,
            activation=nn.ReLU,
            attention_downsample_rate: int = 2,
            skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
            self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ):
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class MLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act=nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
            self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@MODELS.register_module()
class CDPseudoHead(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 num_classes,
                 threshold=0.5,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 align_corners=False,
                 init_cfg=None,
                 ):
        super().__init__(init_cfg)
        self.num_classes = num_classes
        self.threshold = threshold
        self.loss_decode = MODELS.build(loss_decode)
        self.align_corners = align_corners
        self.out_channels = self.num_classes

    def forward(self, x):
        return x

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        # import ipdb; ipdb.set_trace()
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_label = seg_label.squeeze(1)
        loss_decode = self.loss_decode

        loss[loss_decode.loss_name] = loss_decode(
            seg_logits,
            seg_label)
        loss['acc_seg'] = accuracy(
            seg_logits, seg_label)
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        seg_logits = resize(
            input=seg_logits,
            size=batch_img_metas[0]['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits
