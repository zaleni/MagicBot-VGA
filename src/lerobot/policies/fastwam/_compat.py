from __future__ import annotations

import sys
import types
from pathlib import Path


def ensure_local_fastwam_package_tree() -> None:
    base_dir = Path(__file__).resolve().parent
    package_dirs = {
        "lerobot.policies.fastwam.core": base_dir / "core",
        "lerobot.policies.fastwam.core.data": base_dir / "core" / "data",
        "lerobot.policies.fastwam.core.data.lerobot": base_dir / "core" / "data" / "lerobot",
        "lerobot.policies.fastwam.core.data.lerobot.processors": base_dir / "core" / "data" / "lerobot" / "processors",
        "lerobot.policies.fastwam.core.data.lerobot.transforms": base_dir / "core" / "data" / "lerobot" / "transforms",
        "lerobot.policies.fastwam.core.data.lerobot.utils": base_dir / "core" / "data" / "lerobot" / "utils",
        "lerobot.policies.fastwam.core.models": base_dir / "core" / "models",
        "lerobot.policies.fastwam.core.models.wan22": base_dir / "core" / "models" / "wan22",
        "lerobot.policies.fastwam.core.models.wan22.helpers": base_dir / "core" / "models" / "wan22" / "helpers",
        "lerobot.policies.fastwam.core.models.wan22.schedulers": base_dir / "core" / "models" / "wan22" / "schedulers",
        "lerobot.policies.fastwam.core.utils": base_dir / "core" / "utils",
    }

    for module_name, module_dir in package_dirs.items():
        if not module_dir.is_dir():
            raise ModuleNotFoundError(
                f"Expected local FastWAM package directory is missing: {module_dir} ({module_name})"
            )

        module = sys.modules.get(module_name)
        if module is None:
            module = types.ModuleType(module_name)
            sys.modules[module_name] = module

        module.__package__ = module_name
        module.__path__ = [str(module_dir)]
        init_file = module_dir / "__init__.py"
        module.__file__ = str(init_file if init_file.is_file() else module_dir)
