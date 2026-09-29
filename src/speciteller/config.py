"""SpeciTeller integration configuration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeciTellerConfig:
    image: str
    repo_commit: str
    data_checksum: str
    liblinear_version: str


DEFAULT_SPECITELLER_CONFIG = SpeciTellerConfig(
    image="speciteller:py27",
    repo_commit="218cb5a389b3e51d393a76e72714ff65e7e81f47",
    data_checksum="dd70443a77b6fc576e3fff88ddc09e6e0777bea3537465a11f339b9de8105fbe",
    liblinear_version="491c9f1188b97ba70847c70a68be363d186ddf9d",
)