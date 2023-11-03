#!/usr/bin/env bash
source ~/.bashrc
conda activate torch2mmcv2  # torch1mmcv1 torch1mmcv2 torch2mmcv1 torch2mmcv2
export XDG_CACHE_HOME=pretrain_models
cd /mnt/search01/usr/chenkeyan/codes/ovarnet

pip install importlib_metadata
pip install open_clip_torch
pip install peft
pip install wandb --upgrade
pip install openmim
mim install mmengine --upgrade
pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html --upgrade
pip install torchmetrics --upgrade
pip install imagesize



#pip install deepspeed
# pip install anypackage
# yum install which
# source /opt/rh/devtoolset-9/enable
# mim install mmcv>=2.0.0rc4
# TORCH_DISTRIBUTED_DEBUG=DETAIL


case $# in
0)
    tools/dist_train.sh $1
    ;;
1)
    tools/dist_train.sh $1
    ;;
2)
    tools/dist_train.sh $1 $2
    ;;
esac
# TORCH_DISTRIBUTED_DEBUG=DETAIL
#python train.py
#python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env train.py
#python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env train_pipe.py
# juicesync src dst
# juicefs rmr your_dir
# bash /mnt/search01/usr/chenkeyan/codes/ovarnet/qs_run.sh configs/ovarnet/stage1_clipattr_align_coco_vaw.py 4
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH tools/dist_train.sh configs/ovarnet/stage1_clipattr_align_coco_vaw_r16.py 8