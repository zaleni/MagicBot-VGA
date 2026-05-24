# LIBERO-Plus Quick Start

Start the policy server in the `magicbot` environment:

```bash
export CUDA_VISIBLE_DEVICES=0
cd MagicBot-VGA
conda activate magicbot
PORT=8006 \
CHECKPOINT_DIR=/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/outputs/MagicBot_R0/magicbot_r0-pretrained-libero4-2026_05_21_18_03_05/checkpoints/110000/pretrained_model \
LOAD_TEXT_ENCODER=true \
INFER_HORIZON=12 \
bash evaluation/Libero-plus/02_serve_magicbot_liberoplus.sh
```

For `MagicBot_R0`, either point `TEXT_EMBED_CACHE_DIR` at precomputed LIBERO-Plus prompt embeddings, or set `LOAD_TEXT_ENCODER=true` to encode prompts during serving.

Run the evaluator in the `liberoplus` environment:

```bash
export CUDA_VISIBLE_DEVICES=3
cd MagicBot-VGA
conda activate liberoplus
export LIBERO_HOME=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/LIBERO-plus
export PYTHONPATH=$LIBERO_HOME:$PYTHONPATH
WS_URL=ws://127.0.0.1:8009 \
TASK_SUITE_NAME=libero_object \
INFER_HORIZON=12 \
NUM_TRIALS_PER_TASK=1 \
VIDEO_ROOT=$PWD/evaluation/Libero-plus/magicbotr0 \
bash evaluation/Libero-plus/eval.sh
```
## Resume Evaluation

If the evaluation is interrupted, add `TASK_ID=<start_id>` on the inference side. The evaluator will resume from `start_id`.

## LIBERO-Plus Results

To summarize the LIBERO-Plus benchmark results, run `post-process.py`.