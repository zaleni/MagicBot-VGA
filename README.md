# MagicBot-VGA

This repository documents how to evaluate our RoboTwin model
[`zaleni/MagicBot-VGA-Robotwin`](https://huggingface.co/zaleni/MagicBot-VGA-Robotwin)
with the MagicBot-VGA codebase.

Repository:

- https://github.com/zaleni/MagicBot-VGA

Model:

- https://huggingface.co/zaleni/MagicBot-VGA-Robotwin

This README focuses on RoboTwin 2.0 environment preparation and evaluation.

It covers:

- MagicBot environment installation
- RoboTwin evaluation setup
- required external model assets
- single-task evaluation
- 50-task randomized evaluation
- custom task subset evaluation

## 1. Requirements

The codebase is built and tested with:

- Python 3.10
- CUDA 12.8
- PyTorch 2.7.1

We recommend using a Linux machine with NVIDIA GPUs.

## 2. Install the MagicBot Base Environment

Clone the repository:

```bash
git clone https://github.com/zaleni/MagicBot-VGA.git
cd MagicBot-VGA
```

Create a conda environment:

```bash
conda create -y -n magicbot python=3.10
conda activate magicbot
pip install --upgrade pip
```

Install the basic system dependencies used by the codebase:

```bash
conda install -c conda-forge ffmpeg=7.1.1 svt-av1 -y
```

Install PyTorch for CUDA 12.8:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install Python dependencies:

```bash
pip install torchcodec numpy scipy transformers==4.57.1 mediapy loguru pytest omegaconf
pip install -e .
```

## 3. Qwen3-VL Dependency

For `CubeV2`, the recommended dependency is the official Hugging Face `Qwen3-VL`
implementation provided by `transformers>=4.57.0`.

In this repository, `CubeV2` imports Qwen3-VL directly from:

```python
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLTextModel
```

So for standard evaluation, you do not need to patch `transformers` if your environment
already uses a recent enough official version such as `transformers==4.57.1`.

This repo also contains a vendored replacement file under:

```text
src/lerobot/policies/cubev2/transformers_replace/models/qwen3_vl/modeling_qwen3_vl.py
```

That file is best understood as a repo-side override copy. Most users evaluating
`zaleni/MagicBot-VGA-Robotwin` should not need it unless they intentionally want to
reproduce a specific local patched behavior.

## 4. Prepare RoboTwin for Evaluation

This section is specifically for RoboTwin evaluation. If you only want to load the
model or run other parts of the codebase, the extra RoboTwin setup below is not required.

### Option A: initialize the bundled RoboTwin submodule

```bash
git submodule update --init third_party/RoboTwin
```

### Option B: copy an existing RoboTwin checkout

You do not have to download RoboTwin from scratch if you already have a prepared copy.
You can copy it into this repository instead.

The evaluation code assumes RoboTwin is located exactly at:

```text
<repo_root>/third_party/RoboTwin
```

So a valid layout looks like:

```text
MagicBot-VGA/
  evaluation/
  launch/
  src/
  third_party/
    RoboTwin/
```

If your RoboTwin directory already exists elsewhere, either:

- copy it to `third_party/RoboTwin`, or
- create a symlink at `third_party/RoboTwin` pointing to your existing RoboTwin directory

This path requirement comes from the evaluation code, which imports RoboTwin modules
and task configs from `third_party/RoboTwin` directly.

### Install RoboTwin-specific system dependency

RoboTwin rendering requires Vulkan:

```bash
sudo apt install -y libvulkan1 mesa-vulkan-drivers vulkan-tools
```

### Install RoboTwin Python dependencies and assets

```bash
cp evaluation/RoboTwin/requirements.txt third_party/RoboTwin/script/requirements.txt
cd third_party/RoboTwin
bash script/_install.sh
bash script/_download_assets.sh
cd ../../
```

For more RoboTwin installation details, you can also refer to the official documentation:
https://robotwin-platform.github.io/doc/usage/robotwin-install.html

## 5. Prepare External Model Assets

The released checkpoint `zaleni/MagicBot-VGA-Robotwin` is intended to be lightweight.
For RoboTwin action evaluation, you should provide the external backbone/tokenizer assets explicitly.

Recommended values:

- Qwen3-VL backbone and processor: `Qwen/Qwen3-VL-2B-Instruct`
- Cosmos tokenizer: `nvidia/Cosmos-Tokenizer-CI8x8`

You can use either:

- public Hugging Face repo ids
- local directories downloaded in advance

Example for offline/local usage:

```bash
QWEN3_VL_PATH=/path/to/Qwen3-VL-2B-Instruct
COSMOS_TOKENIZER_PATH=/path/to/Cosmos-Tokenizer-CI8x8
```

For standard RoboTwin action evaluation, we recommend disabling DA3 teacher instantiation:

```bash
DISABLE_DA3_TEACHER_FOR_EVAL=true
```

This avoids loading the frozen DA3 teacher during evaluation while keeping the policy architecture compatible.

## 6. Single-Task Evaluation

The most direct way is to call `evaluation/RoboTwin/inference.py` on a single RoboTwin task.

Example: evaluate task `0` (`adjust_bottle`) on `demo_clean`:

```bash
cd third_party/RoboTwin

python ../../evaluation/RoboTwin/inference.py \
  --args.ckpt_path zaleni/MagicBot-VGA-Robotwin \
  --args.video_dir ../../evaluation/RoboTwin/output_magicbot/demo_clean/task_00 \
  --args.task_config demo_clean \
  --args.task_idx 0 \
  --args.action_mode delta \
  --args.stats_key aloha \
  --args.dtype bfloat16 \
  --args.qwen3_vl_pretrained_path Qwen/Qwen3-VL-2B-Instruct \
  --args.qwen3_vl_processor_path Qwen/Qwen3-VL-2B-Instruct \
  --args.cosmos_tokenizer_path_or_name nvidia/Cosmos-Tokenizer-CI8x8 \
  --args.disable_3d_teacher_for_eval
```

If you use local asset directories, replace the public repo ids with your local paths.

Important arguments:

- `--args.ckpt_path`: model repo id or local `pretrained_model` directory
- `--args.task_config`: `demo_clean` or `demo_randomized`
- `--args.task_idx`: task index in `evaluation/RoboTwin/inference.py`
- `--args.action_mode`: usually `delta` for this model
- `--args.stats_key`: usually `aloha` for RoboTwin
- `--args.dtype`: `bfloat16` is recommended on modern GPUs

Outputs are written to `--args.video_dir`, including:

- replay videos
- `summary.json`
- `summary.txt`

## 7. 50-Task Randomized Evaluation

For batch evaluation on RoboTwin randomized tasks, use:

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

Useful environment variables:

- `PRETRAINED_CKPT`: model repo id or local checkpoint directory
- `GPU_IDS`: comma-separated GPU ids, for example `0,1,2,3`
- `MAX_JOBS_PER_GPU`: parallel RoboTwin jobs per GPU
- `TASK_CONFIG`: defaults to `demo_randomized`
- `TEST_NUM`: number of episodes per task
- `DTYPE`: `bfloat16` or `float32`
- `BASE_OUTPUT_PATH`: output root directory

This script writes:

- per-task logs and videos under `tasks/`
- aggregated `summary.json`
- aggregated `summary.txt`

## 8. Evaluate a Continuous Task Range

`eval_randomized_50.sh` supports continuous ranges through:

- `START_TASK_IDX`
- `TASK_COUNT`

Example: evaluate tasks `10` to `19`:

```bash
PRETRAINED_CKPT=zaleni/MagicBot-VGA-Robotwin \
QWEN3_VL_PRETRAINED_PATH=Qwen/Qwen3-VL-2B-Instruct \
QWEN3_VL_PROCESSOR_PATH=Qwen/Qwen3-VL-2B-Instruct \
COSMOS_TOKENIZER_PATH_OR_NAME=nvidia/Cosmos-Tokenizer-CI8x8 \
DISABLE_DA3_TEACHER_FOR_EVAL=true \
START_TASK_IDX=10 \
TASK_COUNT=10 \
bash evaluation/RoboTwin/eval_randomized_50.sh
```

## 9. Evaluate a Custom Sparse Task List

If you want to evaluate a sparse subset such as:

```text
[2, 3, 9, 10, 12, 15, 17, 25, 28, 30, 44]
```

the current batch script does not take a sparse task list directly. The recommended approach is to run a shell loop:

```bash
cd third_party/RoboTwin

TASKS=(2 3 9 10 12 15 17 25 28 30 44)
for t in "${TASKS[@]}"; do
  python ../../evaluation/RoboTwin/inference.py \
    --args.ckpt_path zaleni/MagicBot-VGA-Robotwin \
    --args.video_dir ../../evaluation/RoboTwin/output_magicbot/custom_subset/task_${t} \
    --args.task_config demo_randomized \
    --args.task_idx "${t}" \
    --args.action_mode delta \
    --args.stats_key aloha \
    --args.dtype bfloat16 \
    --args.qwen3_vl_pretrained_path Qwen/Qwen3-VL-2B-Instruct \
    --args.qwen3_vl_processor_path Qwen/Qwen3-VL-2B-Instruct \
    --args.cosmos_tokenizer_path_or_name nvidia/Cosmos-Tokenizer-CI8x8 \
    --args.disable_3d_teacher_for_eval
done
```

## 10. Task Index Reference

Task indices are defined in [`evaluation/RoboTwin/inference.py`](evaluation/RoboTwin/inference.py).

For example:

- `0`: `adjust_bottle`
- `2`: `blocks_ranking_rgb`
- `3`: `blocks_ranking_size`
- `9`: `handover_mic`
- `10`: `hanging_mug`
- `12`: `move_can_pot`
- `15`: `move_stapler_pad`
- `17`: `open_microwave`
- `25`: `place_can_basket`
- `28`: `place_dual_shoes`
- `30`: `place_fan`
- `44`: `stack_blocks_three`

## 11. Common Notes

- `inference.py` can load checkpoints from either a local directory or a Hugging Face repo id.
- If your server cannot access Hugging Face online, download the external assets in advance and pass local paths.
- If you use the lightweight checkpoint release for action evaluation, keeping `--args.disable_3d_teacher_for_eval` enabled is recommended.
- If you want to inspect reconstructed future images during inference, enable `--args.decode_image_flag`, though this is not required for standard RoboTwin scoring.

## 12. Model Link

Released RoboTwin checkpoint:

- https://huggingface.co/zaleni/MagicBot-VGA-Robotwin

## 13. Acknowledgments

MagicBot-VGA is developed on top of the excellent InternVLA framework. Our codebase
started from that foundation and has since been substantially modified and extended
for our own model architecture, training pipeline, and evaluation workflow.

We sincerely thank the [InternVLA](https://github.com/InternRobotics/InternVLA-A1)
authors and contributors for open-sourcing their framework and making follow-up
research and development much easier.

We also thank the following open-source projects:

- [InternVLA](https://github.com/InternRobotics/InternVLA-A1)
- [LeRobot](https://github.com/huggingface/lerobot)
- [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [NVIDIA Cosmos](https://github.com/nvidia-cosmos)
