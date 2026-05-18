from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


DEFAULT_CALIBRATION_BOARD = "calibio-12x9-30mm"
LEGACY_CALIBRATION_BOARD = "legacy-4x5-50mm"


@dataclass(frozen=True)
class CharucoBoardConfig:
    name: str
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    dictionary_name: str
    deprecated: bool = False
    description: str = ""

    @property
    def square_length_mm(self) -> float:
        return self.square_length_m * 1000.0

    @property
    def marker_length_mm(self) -> float:
        return self.marker_length_m * 1000.0

    @property
    def chessboard_corner_count(self) -> int:
        return max(0, (self.squares_x - 1) * (self.squares_y - 1))

    def with_overrides(
        self,
        *,
        squares_x: int | None = None,
        squares_y: int | None = None,
        square_size_mm: float | None = None,
        marker_size_mm: float | None = None,
        dictionary_name: str | None = None,
    ) -> "CharucoBoardConfig":
        overrides: dict[str, Any] = {}
        if squares_x is not None:
            overrides["squares_x"] = int(squares_x)
        if squares_y is not None:
            overrides["squares_y"] = int(squares_y)
        if square_size_mm is not None:
            overrides["square_length_m"] = float(square_size_mm) / 1000.0
        if marker_size_mm is not None:
            overrides["marker_length_m"] = float(marker_size_mm) / 1000.0
        if dictionary_name is not None:
            overrides["dictionary_name"] = str(dictionary_name)
        if overrides:
            overrides["name"] = f"{self.name}+overrides"
            overrides["deprecated"] = False
            overrides["description"] = f"{self.description} with CLI overrides".strip()
        return replace(self, **overrides)


CALIBRATION_BOARD_CONFIGS: dict[str, CharucoBoardConfig] = {
    DEFAULT_CALIBRATION_BOARD: CharucoBoardConfig(
        name=DEFAULT_CALIBRATION_BOARD,
        squares_x=12,
        squares_y=9,
        square_length_m=0.030,
        marker_length_m=0.022,
        dictionary_name="DICT_5X5_250",
        description="Calib.io 12x9 ChArUco board, checker 30 mm, marker 22 mm",
    ),
    LEGACY_CALIBRATION_BOARD: CharucoBoardConfig(
        name=LEGACY_CALIBRATION_BOARD,
        squares_x=4,
        squares_y=5,
        square_length_m=0.050,
        marker_length_m=0.037,
        dictionary_name="DICT_4X4_50",
        deprecated=True,
        description="Deprecated legacy 4x5 ChArUco board",
    ),
}


def available_calibration_boards() -> tuple[str, ...]:
    return tuple(CALIBRATION_BOARD_CONFIGS)


def get_calibration_board_config(name: str | CharucoBoardConfig | None) -> CharucoBoardConfig:
    if isinstance(name, CharucoBoardConfig):
        return name
    if name is None:
        return CALIBRATION_BOARD_CONFIGS[DEFAULT_CALIBRATION_BOARD]
    try:
        return CALIBRATION_BOARD_CONFIGS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown calibration board profile {name!r}. "
            f"Available profiles: {', '.join(available_calibration_boards())}"
        ) from exc


def charuco_board_config_to_metadata(config: CharucoBoardConfig) -> dict[str, Any]:
    metadata = asdict(config)
    metadata["square_length_mm"] = config.square_length_mm
    metadata["marker_length_mm"] = config.marker_length_mm
    metadata["chessboard_corner_count"] = config.chessboard_corner_count
    return metadata


def resolve_aruco_dictionary_id(dictionary_name: str) -> int:
    import cv2

    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"Unsupported cv2.aruco dictionary name: {dictionary_name!r}")
    return int(dictionary_id)


def create_charuco_board(config: CharucoBoardConfig):
    import cv2

    if config.squares_x < 2 or config.squares_y < 2:
        raise ValueError(
            "ChArUco board must have at least 2 squares in each direction. "
            f"Got {config.squares_x}x{config.squares_y}."
        )
    if config.square_length_m <= 0 or config.marker_length_m <= 0:
        raise ValueError(
            "ChArUco board square and marker lengths must be positive. "
            f"square={config.square_length_m}, marker={config.marker_length_m}"
        )
    if config.marker_length_m >= config.square_length_m:
        raise ValueError(
            "ChArUco marker length must be smaller than square length. "
            f"square={config.square_length_m}, marker={config.marker_length_m}"
        )
    dictionary = cv2.aruco.getPredefinedDictionary(
        resolve_aruco_dictionary_id(config.dictionary_name)
    )
    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (config.squares_x, config.squares_y),
            squareLength=config.square_length_m,
            markerLength=config.marker_length_m,
            dictionary=dictionary,
        )
    elif hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(
            config.squares_x,
            config.squares_y,
            config.square_length_m,
            config.marker_length_m,
            dictionary,
        )
    else:
        raise RuntimeError("This OpenCV build does not provide ChArUco board APIs.")
    return dictionary, board


def get_charuco_chessboard_corners(board):
    if hasattr(board, "getChessboardCorners"):
        return board.getChessboardCorners()
    if hasattr(board, "chessboardCorners"):
        return board.chessboardCorners
    raise RuntimeError("Unsupported ChArUco board object: missing chessboard corners.")
