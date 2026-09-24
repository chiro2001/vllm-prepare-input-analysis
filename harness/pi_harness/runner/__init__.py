"""P1：真实代码路径 replay（不挂 NPU 卡）。"""

from .config import RunnerConfig
from .build import build_runner, AttrAudit

__all__ = ["RunnerConfig", "build_runner", "AttrAudit"]
