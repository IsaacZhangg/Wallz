"""WallZero: a full-board, rule-only AlphaZero engine for Quoridor."""

from wallzero.constants import ACTION_SIZE, BOARD_SIZE, INPUT_PLANES
from wallzero.game import State

__all__ = ["ACTION_SIZE", "BOARD_SIZE", "INPUT_PLANES", "State"]
__version__ = "0.1.0"
