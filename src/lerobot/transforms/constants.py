from collections import defaultdict
from collections.abc import Mapping

from lerobot.utils.constants import OBS_STATE, ACTION, OBS_IMAGES, OBS_IMAGE
from .utils import make_bool_mask


MASK_MAPPING = {
    # a1 old
    "piper": make_bool_mask(6, -1, 6, -1),  # split_aloha
    "arx_lift2": make_bool_mask(6, -1, 6, -1), 
    "split_aloha": make_bool_mask(6, -1, 6, -1), 
    "a2d": make_bool_mask(14, -2),  # agibotworld
    "genie1": make_bool_mask(14, -2), 
    "franka": make_bool_mask(7, -1), 
    "frankarobotiq": make_bool_mask(7, -1), 
    # a1 new
    "Franka": make_bool_mask(7, -1), 
    "ARX Lift-2": make_bool_mask(6, -1, 6, -1), 
    "AgileX Split Aloha": make_bool_mask(6, -1, 6, -1), 
    "Genie-1": make_bool_mask(14, -2), 
    "ARX AC One": make_bool_mask(6, -1, 6, -1), 
    # others
    "aloha": make_bool_mask(6, -1, 6, -1), 
    "panda": make_bool_mask(7, ), 
    "egodex_v": make_bool_mask(2),
    # RoboChallenge
    "ALOHA": make_bool_mask(6, -1, 6, -1),
    "UR5": make_bool_mask(6, -1),
    "ARX5": make_bool_mask(6, -1),
    "FRANKA": make_bool_mask(7, -2),
    # custom real robot
    "real_lift2": make_bool_mask(6, -1, 6, -1),
}

LIBERO_FRANKA_MASK = make_bool_mask(7, -1)
MASK_MAPPING["libero_franka"] = LIBERO_FRANKA_MASK


FEATURE_MAPPING = defaultdict(
    lambda : {
        OBS_STATE: ["observation.state"],
        ACTION: ["action"],
    }, 
    a2d={
        OBS_STATE: [
            "observation.states.joint.position", 
            "observation.states.effector.position", 
        ], 
        ACTION: [
            "actions.joint.position", 
            "actions.effector.position", 
        ], 
    }, 
    genie1={
        OBS_STATE: [
            "states.left_joint.position", 
            "states.right_joint.position", 
            "states.left_gripper.position", 
            "states.right_gripper.position", 
        ], 
        ACTION: [
            "actions.left_joint.position", 
            "actions.right_joint.position", 
            "actions.left_gripper.position", 
            "actions.right_gripper.position", 
        ], 
    }, 
    arx_lift2={
        OBS_STATE: [
            "states.left_joint.position", 
            "states.left_gripper.position", 
            "states.right_joint.position", 
            "states.right_gripper.position", 
        ], 
        ACTION: [
            "actions.left_joint.position", 
            "actions.left_gripper.position", 
            "actions.right_joint.position", 
            "actions.right_gripper.position", 
        ], 
    }, 
    piper={
        OBS_STATE: [
            "states.left_joint.position", 
            "states.left_gripper.position", 
            "states.right_joint.position", 
            "states.right_gripper.position", 
        ], 
        ACTION: [
            "actions.left_joint.position", 
            "actions.left_gripper.position", 
            "actions.right_joint.position", 
            "actions.right_gripper.position", 
        ], 
    }, 
    r1lite={
        OBS_STATE: [
            'observation.state.left_arm', 
            'observation.state.right_arm', 
            'observation.state.left_gripper', 
            'observation.state.right_gripper',
        ], 
        ACTION: [
            "action.left_arm", 
            "action.right_arm",
            "action.left_gripper",
            "action.right_gripper",
        ], 
    },
    aloha={
        OBS_STATE: [
            'observation.state',
        ], 
        ACTION: [
            'action',
        ], 
    },
    franka={
        OBS_STATE: [
            "states.joint.position", 
            "states.gripper.position",
        ], 
        ACTION: [
            "actions.joint.position", 
            "actions.gripper.position", 
        ], 
    }, 
    panda={
        OBS_STATE: [
            "observation.state", 
        ], 
        ACTION: [
            "action", 
        ], 
    },
    ALOHA={
        OBS_STATE: [
            "observation.state",
        ],
        ACTION: [
            "action",
        ],
    },
    UR5={
        OBS_STATE: [
            "observation.state",
        ],
        ACTION: [
            "action",
        ],
    },
    ARX5={
        OBS_STATE: [
            "observation.state",
        ],
        ACTION: [
            "action",
        ],
    },
    FRANKA={
        OBS_STATE: [
            "observation.state",
        ],
        ACTION: [
            "action",
        ],
    },
)
# a1 new
FEATURE_MAPPING["Franka"] = {
    OBS_STATE: [
            "states.joint.position", 
            "states.gripper.position",
    ], 
    ACTION: [
        "actions.joint.position", 
        "actions.gripper.position", 
    ], 
}
FEATURE_MAPPING["ARX Lift-2"] = {
    OBS_STATE: [
            "states.left_joint.position", 
            "states.left_gripper.position", 
            "states.right_joint.position", 
            "states.right_gripper.position", 
        ], 
    ACTION: [
        "actions.left_joint.position", 
        "actions.left_gripper.position", 
        "actions.right_joint.position", 
        "actions.right_gripper.position", 
    ], 
}
FEATURE_MAPPING["Genie-1"] = {
    OBS_STATE: [
        "states.left_joint.position", 
        "states.right_joint.position", 
        "states.left_gripper.position", 
        "states.right_gripper.position", 
    ], 
    ACTION: [
        "actions.left_joint.position", 
        "actions.right_joint.position", 
        "actions.left_gripper.position", 
        "actions.right_gripper.position", 
    ], 
}
FEATURE_MAPPING["AgileX Split Aloha"] = {
    OBS_STATE: [
        "states.left_joint.position", 
        "states.left_gripper.position", 
        "states.right_joint.position", 
        "states.right_gripper.position", 
    ], 
    ACTION: [
        "actions.left_joint.position", 
        "actions.left_gripper.position", 
        "actions.right_joint.position", 
        "actions.right_gripper.position", 
    ], 
}
FEATURE_MAPPING["ARX AC One"] = {
    OBS_STATE: [
        "states.left_joint.position", 
        "states.left_gripper.position", 
        "states.right_joint.position", 
        "states.right_gripper.position", 
    ], 
    ACTION: [
        "actions.left_joint.position", 
        "actions.left_gripper.position", 
        "actions.right_joint.position", 
        "actions.right_gripper.position", 
    ], 
}
FEATURE_MAPPING["egodex_v"] = {
    OBS_STATE: [
        "observation.state",
    ],
    ACTION: [
        "action",
    ],
}
FEATURE_MAPPING["libero_franka"] = {
    OBS_STATE: [
        "observation.state",
    ],
    ACTION: [
        "action",
    ],
}
FEATURE_MAPPING["real_lift2"] = {
    OBS_STATE: [
        "observation.state",
    ],
    ACTION: [
        "action",
    ],
}


IMAGE_MAPPING = defaultdict(
    lambda : {
        "observation.image": f"{OBS_IMAGES}.image0", 
    }, 
    arx_lift2={
        "images.rgb.head": f"{OBS_IMAGES}.image0", 
        "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
        "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
    }, 
    piper={
        "images.rgb.head": f"{OBS_IMAGES}.image0", 
        "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
        "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
    },
    genie1={
        "images.rgb.head": f"{OBS_IMAGES}.image0", 
        "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
        "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
    }, 
    a2d={
        "observation.images.head": f"{OBS_IMAGES}.image0", 
        "observation.images.hand_left": f"{OBS_IMAGES}.image1", 
        "observation.images.hand_right": f"{OBS_IMAGES}.image2", 
    }, 
    # todo, make sure what the key names are for franka
    franka={
        "images.rgb.head": f"{OBS_IMAGES}.image0", 
        "images.rgb.hand": f"{OBS_IMAGES}.image1", 
    }, 
    r1lite={
        "observation.images.head_rgb": f"{OBS_IMAGES}.image0", 
        "observation.images.left_wrist_rgb": f"{OBS_IMAGES}.image1", 
        "observation.images.right_wrist_rgb": f"{OBS_IMAGES}.image2", 
    },

    aloha={
        "observation.images.cam_high": f"{OBS_IMAGES}.image0", 
        "observation.images.cam_left_wrist": f"{OBS_IMAGES}.image1", 
        "observation.images.cam_right_wrist": f"{OBS_IMAGES}.image2", 
    },
    panda={
        "observation.images.image": f"{OBS_IMAGES}.image0", 
        "observation.images.image2": f"{OBS_IMAGES}.image1", 
    }
)
# a1 new
IMAGE_MAPPING["Franka"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0", 
    "images.rgb.hand": f"{OBS_IMAGES}.image1", 
}
IMAGE_MAPPING["ARX Lift-2"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0", 
    "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
    "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
}
IMAGE_MAPPING["Genie-1"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0", 
    "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
    "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
}
IMAGE_MAPPING["AgileX Split Aloha"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0", 
    "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
    "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
}
IMAGE_MAPPING["ARX AC One"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0", 
    "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
    "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
}
IMAGE_MAPPING["ALOHA"] = {
    "observation.images.head": f"{OBS_IMAGES}.image0",
    "observation.images.left": f"{OBS_IMAGES}.image1",
    "observation.images.right": f"{OBS_IMAGES}.image2",
}
IMAGE_MAPPING["UR5"] = {
    "observation.images.head": f"{OBS_IMAGES}.image0",
    "observation.images.left": f"{OBS_IMAGES}.image1",
}
IMAGE_MAPPING["ARX5"] = {
    "observation.images.head": f"{OBS_IMAGES}.image0",
    "observation.images.left": f"{OBS_IMAGES}.image1",
    "observation.images.right": f"{OBS_IMAGES}.image2",
}
IMAGE_MAPPING["FRANKA"] = {
    "observation.images.head": f"{OBS_IMAGES}.image0",
    "observation.images.left": f"{OBS_IMAGES}.image1",
    "observation.images.right": f"{OBS_IMAGES}.image2",
}
IMAGE_MAPPING["egodex_v"] = {
    "observation.image": f"{OBS_IMAGES}.image0",
}
IMAGE_MAPPING["libero_franka"] = {
    "observation.images.image": f"{OBS_IMAGES}.image0",
    "observation.images.wrist_image": f"{OBS_IMAGES}.image1",
}
IMAGE_MAPPING["real_lift2"] = {
    "observation.images.head": f"{OBS_IMAGES}.image0",
    "observation.images.left": f"{OBS_IMAGES}.image1",
    "observation.images.right": f"{OBS_IMAGES}.image2",
}


def _feature_key_set(feature_keys):
    if feature_keys is None:
        return None
    if isinstance(feature_keys, Mapping):
        return set(feature_keys.keys())
    return set(feature_keys)


def infer_embodiment_variant(robot_type, feature_keys=None):
    resolved_robot_type = robot_type
    keys = _feature_key_set(feature_keys)

    # LIBERO datasets are tagged as "franka" but use a different flattened
    # feature schema than the other Franka datasets in this codebase.
    if robot_type == "franka" and keys is not None:
        libero_keys = {
            "observation.state",
            "action",
            "observation.images.image",
            "observation.images.wrist_image",
        }
        if libero_keys.issubset(keys):
            resolved_robot_type = "libero_franka"

    return resolved_robot_type


def get_mask_mapping(robot_type, feature_keys=None):
    return MASK_MAPPING[infer_embodiment_variant(robot_type, feature_keys)]


def get_feature_mapping(robot_type, feature_keys=None):
    return FEATURE_MAPPING[infer_embodiment_variant(robot_type, feature_keys)]


def get_image_mapping(robot_type, feature_keys=None):
    return IMAGE_MAPPING[infer_embodiment_variant(robot_type, feature_keys)]
