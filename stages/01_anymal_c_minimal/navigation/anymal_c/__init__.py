# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0.
# ==============================================================================

"""
ANYmal C navigation environment.

Registered MotrixLab task:
    anymal_c_navigation_minimal

This module currently provides a minimal runnable ANYmal C environment.
It is intended for model loading, visualization, action/observation space
verification, and later extension to a full navigation task.
"""

from .cfg import (
    ANYMAL_C_NAVIGATION_MINIMAL_ENV_NAME,
    AnymalCNavigationMinimalCfg,
)
from .env import AnymalCNavigationMinimalEnv

# Short aliases, matching your planned module comment:
# "导出 AnymalCEnvCfg, AnymalCEnv"
AnymalCEnvCfg = AnymalCNavigationMinimalCfg
AnymalCEnv = AnymalCNavigationMinimalEnv

__all__ = [
    "ANYMAL_C_NAVIGATION_MINIMAL_ENV_NAME",
    "AnymalCNavigationMinimalCfg",
    "AnymalCNavigationMinimalEnv",
    "AnymalCEnvCfg",
    "AnymalCEnv",
]
