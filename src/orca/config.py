"""
Configuration objects and YAML loading for the ORCA AI layer.

`ProviderConfig` mirrors the Part 4 rate table (failure/malformed rates
plus $/1M token rates) for a single FakeLLMProvider. `AppConfig` bundles
the primary + fallback provider configs with the client/assistant retry
settings and is the single object threaded through the LLM client, the
triage assistant, and the eval harness.

Uses pathlib throughout so path resolution is identical on Linux, macOS,
and Windows regardless of whether the caller is pytest, `eval/run_eval.py`,
or `scripts/triage_case.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

# src/orca/config.py -> parents[0]=src/orca, [1]=src, [2]=project root
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


@dataclass
class ProviderConfig:
    """Configuration for a single FakeLLMProvider instance and its rate card."""

    name: str
    failure_rate: float
    malformed_rate: float
    input_rate_per_1m: float
    output_rate_per_1m: float
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise ValueError("provider name must not be empty")
        if not (0.0 <= self.failure_rate <= 1.0):
            raise ValueError(f"failure_rate must be in [0.0, 1.0], got {self.failure_rate}")
        if not (0.0 <= self.malformed_rate <= 1.0):
            raise ValueError(f"malformed_rate must be in [0.0, 1.0], got {self.malformed_rate}")
        if self.input_rate_per_1m < 0.0 or self.output_rate_per_1m < 0.0:
            raise ValueError("token rates must be non-negative")


@dataclass
class AppConfig:
    """Top-level, validated configuration for the ORCA AI layer."""

    primary: ProviderConfig
    fallback: ProviderConfig
    max_retries: int = 3
    backoff_base_seconds: float = 0.0
    max_schema_retries: int = 1

    # Hardcoded fallback of Part 4's exact rate table, used only if
    # config/config.yaml cannot be found on disk (e.g. a stripped-down
    # deployment). config/config.yaml is otherwise the source of truth.
    _PART4_DEFAULTS = {
        "primary": {
            "name": "orca-primary",
            "failure_rate": 0.15,
            "malformed_rate": 0.05,
            "input_rate_per_1m": 3.00,
            "output_rate_per_1m": 15.00,
            "seed": 42,
        },
        "fallback": {
            "name": "orca-fallback",
            "failure_rate": 0.02,
            "malformed_rate": 0.25,
            "input_rate_per_1m": 0.50,
            "output_rate_per_1m": 1.50,
            "seed": 4242,
        },
        "max_retries": 3,
        "backoff_base_seconds": 0.05,
        "max_schema_retries": 1,
    }

    @classmethod
    def default(cls) -> "AppConfig":
        """Load config/config.yaml at the project root, or fall back to
        Part 4's hardcoded rate table if that file is missing."""
        return cls.from_yaml()

    @classmethod
    def from_yaml(cls, path: Optional[os.PathLike] = None) -> "AppConfig":
        config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

        if not config_path.exists():
            raw = cls._PART4_DEFAULTS
        else:
            with config_path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            if not raw:
                raise ValueError(f"Config file at {config_path} is empty or invalid.")

        defaults = cls._PART4_DEFAULTS
        primary_raw = {**defaults["primary"], **raw.get("primary", {})}
        fallback_raw = {**defaults["fallback"], **raw.get("fallback", {})}

        return cls(
            primary=ProviderConfig(
                name=primary_raw["name"],
                failure_rate=float(primary_raw["failure_rate"]),
                malformed_rate=float(primary_raw["malformed_rate"]),
                input_rate_per_1m=float(primary_raw["input_rate_per_1m"]),
                output_rate_per_1m=float(primary_raw["output_rate_per_1m"]),
                seed=int(primary_raw["seed"]),
            ),
            fallback=ProviderConfig(
                name=fallback_raw["name"],
                failure_rate=float(fallback_raw["failure_rate"]),
                malformed_rate=float(fallback_raw["malformed_rate"]),
                input_rate_per_1m=float(fallback_raw["input_rate_per_1m"]),
                output_rate_per_1m=float(fallback_raw["output_rate_per_1m"]),
                seed=int(fallback_raw["seed"]),
            ),
            max_retries=int(raw.get("max_retries", defaults["max_retries"])),
            backoff_base_seconds=float(
                raw.get("backoff_base_seconds", defaults["backoff_base_seconds"])
            ),
            max_schema_retries=int(raw.get("max_schema_retries", defaults["max_schema_retries"])),
        )