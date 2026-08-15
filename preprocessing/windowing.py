"""Windowing placeholder"""
"""
windowing.py

Research-grade CT Windowing Module
Supports multiple CT window presets.

Author: LungCancerAI
"""



from dataclasses import dataclass
from enum import Enum

import numpy as np


class WindowType(Enum):

    LUNG = "lung"

    MEDIASTINAL = "mediastinal"

    BONE = "bone"

    CUSTOM = "custom"


@dataclass
class Window:

    level: float

    width: float


WINDOW_PRESETS = {

    WindowType.LUNG:
        Window(level=-600, width=1500),

    WindowType.MEDIASTINAL:
        Window(level=40, width=400),

    WindowType.BONE:
        Window(level=400, width=1800)

}


class CTWindowing:

    def __init__(

        self,

        preset: WindowType = WindowType.LUNG,

        custom_level=None,

        custom_width=None

    ):

        if preset == WindowType.CUSTOM:

            if custom_level is None or custom_width is None:

                raise ValueError(
                    "Custom window requires level and width."
                )

            self.window = Window(
                custom_level,
                custom_width
            )

        else:

            self.window = WINDOW_PRESETS[preset]

    def apply(

        self,

        volume: np.ndarray

    ) -> np.ndarray:

        level = self.window.level

        width = self.window.width

        lower = level - width / 2

        upper = level + width / 2

        volume = np.clip(
            volume,
            lower,
            upper
        )

        volume = (
            volume - lower
        ) / (upper - lower)

        volume = volume.astype(np.float32)

        return volume

    def get_parameters(self):

        return {

            "level": self.window.level,

            "width": self.window.width

        }