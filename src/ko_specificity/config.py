"""Frozen identity and runtime configuration for Ko et al. (2019)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KoSpecificityConfig:
    image: str
    model_id: str
    model_version: str
    upstream_commit: str
    glove_volume: str
    project_scoring_authorized: bool = False
    score_min: float = 0.0
    score_max: float = 1.0
    score_direction: str = "higher_is_more_specific"


DEFAULT_KO_CONFIG = KoSpecificityConfig(
    image="spectech-ko-specificity:36f8e835-py36-torch100-cpu",
    model_id="ko2019_se_ad_mean_std",
    model_version="ko2019-postpublication-36f8e835-cpu-candidate",
    upstream_commit="36f8e835e9dc6087d5b6763accf302db175947b1",
    glove_volume="spectech-ko-glove-840b-v1",
)
