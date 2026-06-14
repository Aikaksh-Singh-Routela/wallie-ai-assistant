"""Wallie agent control — synthetic input + skills to actually play the game.

Vision sets a high-level skill (slow loop); the controller executes it continuously
(fast loop) until the next decision arrives. Dependency-free (ctypes SendInput).
"""
from .input_controller import InputController

__all__ = ["InputController"]
