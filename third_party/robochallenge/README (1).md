# RoboChallenge Table30 v2 Dataset

## Tasks and Embodiments

The dataset includes 30 diverse manipulation tasks (Table30 v2) across 4 embodiments:

### Available Tasks

- `put_the_books_back` - Place the books back onto the bookshelf.
- `tie_a_knot` - Tie a knot with the string on the table.
- `stamp_positioning` - Stamp the signature area on the paper.
- `tidy_up_the_makeup_table` - Sort and organize the cosmetics on the table.
- `paint_jam` - Spread the bread with jam.
- `pack_the_items` - Box up the tablet and its accessories.
- `wrap_with_a_soft_cloth` - Bundle the objects together using the cloth on the table.
- `put_in_pen_container` - Put the pens on the desk into the pen holder.
- `put_the_pencil_case_into_the_schoolbag` - Put the pencil case into the backpack.
- `put_the_shoes_back` - Pair the two pairs of shoes on the desk and place them on the shoe rack
- `untie_the_shoelaces` - Remove the laces from the shoes, then place them on the table.
- `scoop_with_a_small_spoon` - Scoop beans and place them into the empty bowl.
- `wipe_the_blackboard` - Wipe the balckboard clean.
- `lint_roller_remove_dirt` - Use a lint remover to clean the debris on the clothe.
- `turn_on_the_light_switch` - Turn on the lamp.
- `hold_the_tray_with_both_hands` - Place the ball on the desk onto the small tray, and then move it to the large tray.
- `fold_the_clothes` - Fold the T-shirts and stack them neatly in the upper-left corner of the table.
- `pack_the_toothbrush_holder` - Put the toothbrush and toothpaste into the toiletries case in sequence, close the case, and then place it into the basket.
- `place_objects_into_desk_drawer` - Open the drawer, put the bottle opener inside, and close the drawer.
- `sweep_the_trash` - Sweep the trash on the table into the dustpan.
- `arrange_flowers` - Put the 4 flowers into the vase.
- `press_the_button` - Press the buttons in the following sequence: pink, blue, green, and then yellow.
- `pick_out_the_green_blocks` - Find all the green blocks and put them into the basket.
- `hang_the_cup` - Hang the cup on the rack.
- `water_the_flowers` - Water the potted plants.
- `wipe_the_table` - Wipe the stains off the desk with a rag.
- `arrange_fruits` - Arrange the fruit in the basket.
- `shred_paper` - Put the paper into the shredder.
- `item_classification` - Place the stationery in the yellow box and the electronics in the blue box.
- `stack_bowls` - Put the blue bowl into the beige bowl, and put the green bowl into the blue bowl.

### Embodiments

- **ARX5** - Single-arm with triple camera setup (wrist + global + right-side views)
- **UR5** - Single-arm with dual camera setup (wrist + global views)
- **ALOHA** - Dual-arm with triple wrist camera setup (left wrist + right wrist + global views)
- **DOS-W1** - Dual-arm with triple wrist camera setup (left wrist + right wrist + global views)

## Dataset Structure

### Hierarchy

The dataset is organized by tasks, with each task containing multiple demonstration episodes:

```
.
├── <task_name>/                    # e.g., arrange_the_flowers, fold_t_shirt
│   ├── task_desc.json              # Task description
│   ├── meta/                       # Task-level metadata
│   │   ├── task_info.json         
│   └── data/                       # Episode data
│       ├── episode_000000/         # Individual episode
│       │   ├── meta/
│       │   │   └── episode_meta.json    # Episode metadata
│       │   ├── states/
│       │   │   # for single-arm (ARX5, UR5)
│       │   │   ├── states.jsonl         # Single-arm robot states
│       │   │   # for dual-arm (ALOHA, DOS-W1)
│       │   │   ├── left_states.jsonl    # Left arm states
│       │   │   └── right_states.jsonl   # Right arm states
│       │   └── videos/
│       │       # Video configurations varies by robot model:
│       │       # ARX5
│       │       ├── cam_arm_rgb.mp4       # Wrist view 
│       │       ├── cam_global_rgb.mp4    # Global view 
│       │       └── cam_side_rgb.mp4      # Side view
│       │       # UR5
│       │       ├── cam_global_rgb.mp4    # Global view 
│       │       └── cam_arm_rgb.mp4       # Wrist view
│       │       # ALOHA
│       │       ├── cam_high_rgb.mp4            # Global view 
│       │       ├── cam_left_wrist_rgb.mp4      # Left wrist view
│       │       └── cam_right_wrist_rgb.mp4     # Right wrist view
│       │       # DOS-W1
│       │       ├── cam_high_rgb.mp4            # Global view 
│       │       ├── cam_left_wrist_rgb.mp4      # Left wrist view
│       │       └── cam_right_wrist_rgb.mp4     # Right wrist view
│       ├── episode_000001/
│       └── ...
├── convert_to_lerobot.py           # Conversion script
└── README.md
```

### Metadata Schema

`task_info.json`

```json
{
  "task_desc": {
    "task_name": "arrange_flowers",                 // Task identifier
    "prompt": "Put the 4 flowers into the vase.",
    "description": "...",
    "scoring": "...",                               // Scoring criteria
    "task_tag": [                                   // Task characteristics
      "repeated",
      "single-arm",
      "ARX5",
      "precise3d"
    ]
  },
  "video_info": {
    "fps": 30,                                      // Video frame rate
    "ext": "mp4",                                   // Video format
    "encoding": {
      "vcodec": "libx264",                          // Video codec
      "pix_fmt": "yuv420p"                          // Pixel format
    }
  }
}
```

`episode_meta.json`

```json
{
  "start_time": 1750405586.3430033,                 // Unix timestamp (start)
  "end_time": 1750405642.5247612,                   // Unix timestamp (end)
  "frames": 1672,                                   // Total video frames
  "robot_id": "rc_arx5_5",                          // Robot identifier
  "features": {
    "cam_global": {                                 // Camera name info
      "intrinsics": [],                             // Intrinsics
      "extrinsics": {                               // Extrinsics
        "arms": {
          "arm": []                                 // Extrinsic relative to arm
        }
      }
    }
  }
}
```

### Robot States Schema

Each episode contains states data stored in JSONL format. Depending on the embodiment, the structure differs slightly:

- **Single-arm robots (ARX5, UR5)** → `states.jsonl`
- **Dual-arm robots (ALOHA, DOS-W1)** → `left_states.jsonl` and `right_states.jsonl`

Each file records the robot’s proprioceptive signals per frame, including joint angles,
end-effector poses, gripper states, and timestamps. The exact field definitions and coordinate conventions vary by
platform,
as summarized below.

#### ARX5

|    Data Name     |     Data Key     | Shape |                                                                            Semantics                                                                            |
|:----------------:|:----------------:|:-----:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  Joint control   | joint_positions  | (6,)  |                                                   Joint angle (in radians) from the base to the end effector.                                                   |
|  Joint velocity  | joint_velocities | (6,)  |                                                                        Speed of 6 joint.                                                                        |
|   Joint effort   |     efforts      | (7,)  |                                      Effort of 6 joints and gripper. (Provided by official API, precision not guaranteed.                                       |
|   Pose control   |   ee_positions   | (7,)  | End effector pose (tx, ty, tz, rx, ry, rz, rw), where (tx, ty, tz) is relative position from the arm base coordinate , (rx, ry, rz, rw) is quaternion rotation. |
| Gripper control  |  gripper_width   | (1,)  |                                                           Actual gripper width measurement in meter.                                                            | 
| Gripper velocity | gripper_velocity | (1,)  |                                                                        Speed of gripper.                                                                        |
|    Time stamp    |    timestamp     | (1,)  |                                                    Floating point timestamp (in milliseconds) of each frame.                                                    |

#### UR5

|    Data Name    |    Data Key     | Shape |                                                                            Semantics                                                                            |
|:---------------:|:---------------:|:-----:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  Joint control  | joint_positions | (6,)  |                                                   Joint angle (in radians) from the base to the end effector.                                                   |
|  Pose control   |  ee_positions   | (7,)  | End effector pose (tx, ty, tz, rx, ry, rz, rw), where (tx, ty, tz) is relative position from the arm base coordinate , (rx, ry, rz, rw) is quaternion rotation. |
| Gripper control |  gripper_width  | (1,)  |                                                           Actual gripper width measurement in meter.                                                            | 
|   Time stamp    |    timestamp    | (1,)  |                                                    Floating point timestamp (in milliseconds) of each frame.                                                    | 

#### DOS-W1

|    Data Name    |    Data Key     | Shape |                                                                            Semantics                                                                            |
|:---------------:|:---------------:|:-----:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  Joint control  | joint_positions | (6,)  |                                                   Joint angle (in radians) from the base to the end effector.                                                   |
|  Pose control   |  ee_positions   | (7,)  | End effector pose (tx, ty, tz, rx, ry, rz, rw), where (tx, ty, tz) is relative position from the arm base coordinate , (rx, ry, rz, rw) is quaternion rotation. |
| Gripper control |  gripper_width  | (1,)  |                                                           Actual gripper width measurement in meter.                                                            | 
|   Time stamp    |    timestamp    | (1,)  |                                                    Floating point timestamp (in milliseconds) of each frame.                                                    | 

#### ALOHA

|      Data Name       |     Data Key     | Shape |                                                                            Semantics                                                                            |
|:--------------------:|:----------------:|:-----:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|    Joint control     | joint_positions  | (6,)  |                                               Puppet joint angle (in radians) from the base to the end effector.                                                |
|    Joint velocity    | joint_velocities | (7,)  |                                                                        Speed of 6 joint.                                                                        |
|   Gripper control    |  gripper_width   | (1,)  |                                                           Actual gripper width measurement in meter.                                                            | 
|     Pose control     |   ee_positions   | (7,)  | End effector pose (tx, ty, tz, rx, ry, rz, rw), where (tx, ty, tz) is relative position from the arm base coordinate , (rx, ry, rz, rw) is quaternion rotation. |
|     Joint effort     |     efforts      | (7,)  |                                      Effort of 6 joints and gripper.  (Provided by official API, precision not guaranteed.                                      |
|      Time stamp      |    timestamp     | (1,)  |                                                    Floating point timestamp (in mileseconds) of each frame.                                                     |

## Convert to LeRobot

While you can implement a custom Dataset class to read RoboChallenge data directly, **we strongly recommend converting
to LeRobot format** to take advantage of [LeRobot](https://github.com/huggingface/lerobot)'s comprehensive data
processing and loading utilities.

The example script **`convert_to_lerobot.py`** converts **ARX5** data to the LeRobot dataset as a example. For other
robot embodiments (UR5, ALOHA, DOS-W1), you can adapt the script accordingly.

### Prerequisites

- Python 3.9+ with the following packages:
    - `lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`
    - `opencv-python`
    - `numpy`
- Configure `$HF_LEROBOT_HOME`  (defaults to `~/.cache/huggingface/lerobot` if unset).

```bash
pip install git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 opencv-python numpy
export HF_LEROBOT_HOME="/path/to/lerobot_home"
```

### Usage

Run the converter from the repository root (or provide an absolute path):

```bash
python convert_to_lerobot.py \
  --repo-name example_repo \
  --raw-dataset /path/to/example_dataset \
  --frame-interval 1 
```

### Output

- Frames and metadata are saved to `$HF_LEROBOT_HOME/<repo-name>`.
- At the end, the script calls `dataset.consolidate(run_compute_stats=False)`. If you require aggregated statistics, run
  it with `run_compute_stats=True` or execute a separate stats job.
