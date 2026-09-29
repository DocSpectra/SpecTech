"""Adapter for the exact Ko, Durrett, and Li (2019) implementation."""

from src.ko_specificity.config import DEFAULT_KO_CONFIG, KoSpecificityConfig
from src.ko_specificity.runner import run_ko_specificity

__all__ = ["DEFAULT_KO_CONFIG", "KoSpecificityConfig", "run_ko_specificity"]
