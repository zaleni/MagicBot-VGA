# RoboTwin Evaluation

This document explains how to evaluate the RoboTwin checkpoint
[`zaleni/MagicBot-VGA-Robotwin`](https://huggingface.co/zaleni/MagicBot-VGA-Robotwin)
with this repository.

Unless a command explicitly changes directory, run commands from the repository
root.

[![Model](https://img.shields.io/badge/Model-HuggingFace-FFD21E?logo=huggingface&logoColor=000000)](https://huggingface.co/zaleni/MagicBot-VGA-Robotwin)

## Contents

- RoboTwin 2.0 environment setup
- required external model assets
- single-task evaluation
- 50-task randomized evaluation
- CVPR 2026 RoboTwin Track 11-task evaluation
- submission package generation

## 1. Requirements

The codebase is built and tested with:

- Python 3.10
- CUDA 12.8
- PyTorch 2.7.1

Use a Linux machine with NVIDIA GPUs.

## 2. Base Environment

Clone the repository and create the Python environment:

```bash
git clone https://github.com/zaleni/MagicBot-VGA.git
cd MagicBot-VGA

conda create -y -n magicbot python=3.10
conda activate magicbot
pip install --upgrade pip
```

Install the basic system and Python dependencies:

```bash
conda install -c conda-forge ffmpeg=7.1.1 svt-av1 -y

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

pip install torchcodec numpy scipy transformers==4.57.1 mediapy loguru pytest omegaconf
pip install -e .
```

Optional serving and visualization dependencies:

```bash
pip install tyro matplotlib mediapy websockets msgpack
```

## 3. Patch Qwen3-VL

For CubeV2, install `transformers==4.57.1` first, then patch the installed
Qwen3-VL implementation with the repository copy:

```bash
TRANSFORMERS_DIR=${CONDA_PREFIX}/lib/python3.10/site-packages/transformers/
cp -r src/lerobot/policies/cubev2/transformers_replace/models ${TRANSFORMERS_DIR}
```

CubeV2 imports Qwen3-VL directly from the installed `transformers` package. The
patched file keeps cached decoding compatible with the custom CubeV2 attention
flow used during inference.

## 4. Prepare RoboTwin

### Option A: initialize the bundled RoboTwin submodule

```bash
git submodule update --init third_party/RoboTwin
```

### Option B: copy an existing RoboTwin checkout

The evaluation code assumes RoboTwin is available at:

```text
<repo_root>/third_party/RoboTwin
```

If your RoboTwin directory already exists elsewhere, copy it to
`third_party/RoboTwin` or create a symlink at that path.

### Install RoboTwin dependencies and assets

RoboTwin rendering requires Vulkan:

```bash
sudo apt install -y libvulkan1 mesa-vulkan-drivers vulkan-tools
```

Install RoboTwin Python dependencies and assets:

```bash
cp evaluation/RoboTwin/requirements.txt third_party/RoboTwin/script/requirements.txt
cd third_party/RoboTwin
bash script/_install.sh
bash script/_download_assets.sh
cd ../../
```

Official RoboTwin installation details:
https://robotwin-platform.github.io/doc/usage/robotwin-install.html

## 5. External Model Assets

The released checkpoint is intentionally lightweight. For RoboTwin action
evaluation, provide the external backbone/tokenizer assets explicitly.

Recommended values:

- Qwen3-VL backbone and processor: `Qwen/Qwen3-VL-2B-Instruct`
- Cosmos tokenizer: `nvidia/Cosmos-Tokenizer-CI8x8`

You may use public Hugging Face repo ids or local directories downloaded in
advance:

```bash
QWEN3_VL_PATH=/path/to/Qwen3-VL-2B-Instruct
COSMOS_TOKENIZER_PATH=/path/to/Cosmos-Tokenizer-CI8x8
```

For standard RoboTwin action evaluation, disable DA3 teacher instantiation:

```bash
DISABLE_DA3_TEACHER_FOR_EVAL=true
```

This avoids loading the frozen DA3 teacher during evaluation while keeping the
policy architecture compatible.

## 6. Single-Task Evaluation

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

If you use local asset directories, replace the public repo ids with local
paths.

Important arguments:

- `--args.ckpt_path`: model repo id or local `pretrained_model` directory
- `--args.task_config`: `demo_clean` or `demo_randomized`
- `--args.task_idx`: task index in `evaluation/RoboTwin/inference.py`
- `--args.action_mode`: usually `delta` for this model
- `--args.stats_key`: usually `aloha` for RoboTwin
- `--args.dtype`: `bfloat16` is recommended on modern GPUs

Outputs are written to `--args.video_dir`, including replay videos,
`summary.json`, and `summary.txt`.

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

This script writes per-task logs/videos under `tasks/`, plus aggregated
`summary.json` and `summary.txt`.

## 8. Continuous Task Range

`eval_randomized_50.sh` supports continuous ranges through `START_TASK_IDX` and
`TASK_COUNT`.

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

## 9. CVPR 2026 RoboTwin Track 11-Task Subset

For the Hugging Face leaderboard
[`open-gigaai/CVPR-2026-RoboTwin-Track-LeaderBoard`](https://huggingface.co/spaces/open-gigaai/CVPR-2026-RoboTwin-Track-LeaderBoard),
we use the following 11-task subset:

```text
[2, 3, 9, 10, 12, 15, 17, 25, 28, 30, 44]
```

The exact task names in `evaluation/RoboTwin/inference.py` are:

- `blocks_ranking_rgb`
- `blocks_ranking_size`
- `handover_mic`
- `hanging_mug`
- `move_can_pot`
- `move_stapler_pad`
- `open_microwave`
- `place_can_basket`
- `place_dual_shoes`
- `place_fan`
- `stack_blocks_three`

The batch script does not take a sparse task list directly, so use a shell loop:

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

This produces one output directory per task, each containing replay videos plus
`summary.json` and `summary.txt`.

## 10. Submission Package

After a randomized evaluation run, convert the 11 tasks into a submission-style
folder with:

```bash
python util_scripts/package_robotwin_submission.py \
  --run /path/to/output_randomized_50/<run_name>/summary.txt \
  --dst /path/to/output_randomized_50/<run_name>/submission_package \
  --overwrite
```

If you also want to bundle a policy folder, add:

```bash
  --policy-dir /path/to/policy/Your_Policy
```

The packaging script will:

- create `submission_package/<task_name>/episode0.mp4`, `episode1.mp4`, ...
- preserve the 11-task ordering by task index
- write `package_manifest.txt`
- write `selected_task_summary.json`
- write `selected_task_summary.txt`

The selected-task summary files include per-task success rates/counts, average
task success rate across the 11 tasks, and overall episode success rate.

## 11. Task Index Reference

Task indices are defined in [inference.py](inference.py).

Common indices:

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

## 12. Notes

- `inference.py` can load checkpoints from either a local directory or a Hugging
  Face repo id.
- If your server cannot access Hugging Face online, download external assets in
  advance and pass local paths.
- For the lightweight checkpoint release, keeping
  `--args.disable_3d_teacher_for_eval` enabled is recommended.
- To inspect reconstructed future images during inference, enable
  `--args.decode_image_flag`; this is not required for standard scoring.

## Model Link

Released RoboTwin checkpoint:

- https://huggingface.co/zaleni/MagicBot-VGA-Robotwin
