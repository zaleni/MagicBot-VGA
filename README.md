# MagicBot-VGA / CubeV2

MagicBot-VGA is a robot learning codebase for VLA policy training, simulation
evaluation, and real-robot deployment. This repository contains the CubeV2
policy family, MagicBot_R0 experiments, RoboTwin/Libero evaluation helpers, and
real-robot serving clients for Lift2 and Piper.

[![Repository](https://img.shields.io/badge/Repository-GitHub-181717?logo=github)](https://github.com/zaleni/MagicBot-VGA)
[![RoboTwin Model](https://img.shields.io/badge/RoboTwin%20Model-HuggingFace-FFD21E?logo=huggingface&logoColor=000000)](https://huggingface.co/zaleni/MagicBot-VGA-Robotwin)

## What Is Here

- `src/lerobot/policies/cubev2`: CubeV2 policy, transforms, and model code.
- `src/lerobot/policies/MagicBot_R0`: MagicBot_R0 policy and dataset pipeline.
- `launch/`: training and finetuning entrypoints.
- `evaluation/RoboTwin`: RoboTwin 2.0 evaluation workflow.
- `evaluation/Real_Piper`: sync real-robot Piper serving and ROS1 client.
- `evaluation/Real_Lift2`: Lift2 real-robot serving and inference runtime.
- `evaluation/Libero`: Libero evaluation helpers.
- `util_scripts/`: dataset conversion, norm-stat computation, checkpoint
  repacking, and submission packaging utilities.

## Quick Setup

The core environment is tested with Python 3.10, CUDA 12.8, and PyTorch 2.7.1.

```bash
conda create -y -n magicbot python=3.10
conda activate magicbot
pip install --upgrade pip

conda install -c conda-forge ffmpeg=7.1.1 svt-av1 -y

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

pip install torchcodec numpy scipy transformers==4.57.1 mediapy loguru pytest omegaconf h5py
pip install -e .
```

For real-robot serving, also install:

```bash
pip install tyro matplotlib mediapy websockets msgpack
```

## Qwen3-VL Patch

CubeV2 uses a patched Qwen3-VL implementation for cached inference. After
installing `transformers==4.57.1`, copy the repository patch into the installed
package:

```bash
TRANSFORMERS_DIR=${CONDA_PREFIX}/lib/python3.10/site-packages/transformers/
cp -r src/lerobot/policies/cubev2/transformers_replace/models ${TRANSFORMERS_DIR}
```

## Main Workflows

### RoboTwin Evaluation

The detailed RoboTwin guide now lives in
[evaluation/RoboTwin/README.md](evaluation/RoboTwin/README.md).

Typical batch evaluation:

```bash
PRETRAINED_CKPT=zaleni/MagicBot-VGA-Robotwin \
QWEN3_VL_PRETRAINED_PATH=Qwen/Qwen3-VL-2B-Instruct \
QWEN3_VL_PROCESSOR_PATH=Qwen/Qwen3-VL-2B-Instruct \
COSMOS_TOKENIZER_PATH_OR_NAME=nvidia/Cosmos-Tokenizer-CI8x8 \
DISABLE_DA3_TEACHER_FOR_EVAL=true \
GPU_IDS=0,1 \
MAX_JOBS_PER_GPU=2 \
bash evaluation/RoboTwin/eval_randomized_50.sh
```

### Real Piper Deployment

Piper deployment is documented in
[evaluation/Real_Piper/README.md](evaluation/Real_Piper/README.md).

Useful launch notes:

- [CubeV2 Piper startup note](inference_magicbot_piper.md)
- [MagicBot_R0 Piper startup reference](inference_magicbot_r0_piper.md)

The Piper client supports sync inference, 7D `real_piper` state/action checks,
and Enter-triggered return-to-init/restart when `INIT_JOINT_POSITION` and
`MANUAL_RESET=true` are set.

### Real Lift2 Deployment

Lift2 deployment docs live in
[evaluation/Real_Lift2/README.md](evaluation/Real_Lift2/README.md).

The real-robot server code is shared by Lift2 and Piper where possible, while
robot-side clients stay platform-specific.

### Libero Evaluation

See [evaluation/Libero/README.md](evaluation/Libero/README.md).

## Training Entrypoints

Common CubeV2 launch scripts:

- `launch/cubev2/cubev2_pretrain.sh`
- `launch/cubev2/cubev2_finetune.sh`
- `launch/cubev2/cubev2_finetune_robotwin.sh`
- `launch/cubev2/cubev2_finetune_real_piper.sh`
- `launch/cubev2/cubev2_finetune_real_lift2.sh`

MagicBot_R0 launch scripts:

- `launch/magicbot_r0/magicbot_r0_pretrain.sh`
- `launch/magicbot_r0/magicbot_r0_finetune_robotwin.sh`
- `launch/magicbot_r0/magicbot_r0_finetune_real_piper.sh`
- `launch/magicbot_r0/magicbot_r0_finetune_real_lift2.sh`

Norm-stat utilities for real-robot delta-action models are under
`launch/compute_norm/`.

## Tutorials

- [Installation](tutorials/installation.md)
- [Finetune on LeRobot v2.1 dataset](tutorials/finetune_on_lerobot_v21_dataset.md)
- [Finetune InternVLA-A1 with RoboTwin](tutorials/finetune_internvla_a1_with_robotwin.md)
- [Pretrain InternVLA-A1 with InterData A1](tutorials/pretrain_internvla_a1_with_interndata_a1.md)

## External Assets

Depending on the model path, you may need local copies or Hugging Face repo ids
for:

- Qwen3-VL backbone/processor, for example `Qwen/Qwen3-VL-2B-Instruct`
- Cosmos tokenizer, for example `nvidia/Cosmos-Tokenizer-CI8x8`
- DA3 teacher assets when training or when evaluation explicitly enables them
- MagicBot_R0 Wan/T5/VAE and ActionDiT/Future3D assets

For standard RoboTwin action evaluation with the released lightweight checkpoint,
passing `DISABLE_DA3_TEACHER_FOR_EVAL=true` is recommended.

## Acknowledgments

MagicBot-VGA started from the excellent
[InternVLA](https://github.com/InternRobotics/InternVLA-A1) framework and has
since been extended for CubeV2, MagicBot_R0, real-robot deployment, and multiple
evaluation workflows.

We also thank these open-source projects:

- [LeRobot](https://github.com/huggingface/lerobot)
- [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [NVIDIA Cosmos](https://github.com/nvidia-cosmos)
