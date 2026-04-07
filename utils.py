"""
Shared utilities for the Qwen3-4B tool-use fine-tuning pipeline.

Provides:
  - YAML config loading with env-var override
  - System info logging
  - WandB helper for deterministic run IDs
  - Graceful shutdown handler
"""

from __future__ import annotations

import datetime
import hashlib
import os
import platform
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
_env_path = _PROJECT_ROOT / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    # Also try parent dir (in case running from a sub-folder)
    _parent_env = _PROJECT_ROOT.parent / ".env"
    if _parent_env.exists():
        load_dotenv(_parent_env)


# ═══════════════════════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file and return it as a dict.

    Environment variables in string values are expanded using
    ``os.path.expandvars`` so ``$WANDB_PROJECT`` works in YAML values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    # Expand env vars in string values (one level deep)
    for key, val in cfg.items():
        if isinstance(val, str):
            cfg[key] = os.path.expandvars(val)
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# System info
# ═══════════════════════════════════════════════════════════════════════════════

def log_system_info() -> Dict[str, str]:
    """Collect and print system information. Returns the info dict."""
    import torch

    info: Dict[str, str] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["gpu_model"] = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_mem
        info["gpu_vram_gb"] = f"{vram_bytes / 1024**3:.1f}"
        info["gpu_count"] = str(torch.cuda.device_count())

    # Optional package versions
    for pkg in ("unsloth", "trl", "peft", "transformers", "accelerate",
                "bitsandbytes", "liger_kernel", "vllm", "wandb"):
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "installed")
        except ImportError:
            info[f"{pkg}_version"] = "not installed"

    # Pretty print
    print("\n" + "═" * 60)
    print("  SYSTEM INFO")
    print("═" * 60)
    for k, v in info.items():
        print(f"  {k:30s} : {v}")
    print("═" * 60 + "\n")

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# WandB helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_wandb_run_id(experiment_name: str, extra: str = "") -> str:
    """Generate a deterministic WandB run ID from experiment name + extra info.

    This allows crashed runs to be resumed without creating duplicate entries.
    """
    seed_str = f"{experiment_name}-{extra}"
    return hashlib.md5(seed_str.encode()).hexdigest()[:12]


def init_wandb(
    experiment_name: str,
    config: Dict[str, Any],
    extra_id: str = "",
    tags: Optional[list[str]] = None,
) -> Any:
    """Initialize WandB with deterministic run ID and resume support."""
    import wandb

    project = os.environ.get("WANDB_PROJECT", "grpo_qwen_bfcl")
    entity = os.environ.get("WANDB_ENTITY", "soham-lmao")
    run_id = get_wandb_run_id(experiment_name, extra_id)

    run = wandb.init(
        project=project,
        entity=entity,
        name=experiment_name,
        id=run_id,
        config=config,
        resume="allow",
        tags=tags or [],
    )
    return run


# ═══════════════════════════════════════════════════════════════════════════════
# Git commit hash
# ═══════════════════════════════════════════════════════════════════════════════

def get_git_commit_hash() -> str:
    """Return the current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# Graceful shutdown
# ═══════════════════════════════════════════════════════════════════════════════

class GracefulShutdown:
    """Context manager that catches SIGINT / KeyboardInterrupt and calls a
    user-provided save callback before exiting cleanly.

    Usage::

        def save_fn():
            trainer.save_model("emergency_checkpoint")
            wandb.finish()

        with GracefulShutdown(save_fn):
            trainer.train()
    """

    def __init__(self, save_callback: Optional[callable] = None) -> None:
        self._save_callback = save_callback
        self._interrupted = False
        self._original_handler = None

    def __enter__(self) -> "GracefulShutdown":
        self._original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)
        return self

    def _handler(self, signum: int, frame: Any) -> None:
        if self._interrupted:
            # Second interrupt → hard exit
            sys.exit(1)
        self._interrupted = True
        print("\n⚠️  Interrupt received — saving checkpoint and cleaning up…")
        if self._save_callback:
            try:
                self._save_callback()
            except Exception as e:
                print(f"Error during emergency save: {e}")

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        signal.signal(signal.SIGINT, self._original_handler or signal.SIG_DFL)
        if isinstance(exc_val, KeyboardInterrupt):
            print("Exiting cleanly after interrupt.")
            return True  # suppress the exception
        return False

    @property
    def interrupted(self) -> bool:
        return self._interrupted


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset caching helper
# ═══════════════════════════════════════════════════════════════════════════════

def load_or_download_dataset(
    dataset_name: str,
    cache_dir: str | Path,
    **load_kwargs: Any,
) -> Any:
    """Load dataset from local cache if available, else download and cache.

    Uses ``load_from_disk`` for cached datasets and ``load_dataset`` +
    ``save_to_disk`` for fresh downloads.
    """
    from datasets import load_dataset, load_from_disk

    cache_dir = Path(cache_dir)
    if cache_dir.exists() and any(cache_dir.iterdir()):
        print(f"Loading cached dataset from {cache_dir}")
        return load_from_disk(str(cache_dir))

    print(f"Downloading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, **load_kwargs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache_dir))
    print(f"Saved dataset to {cache_dir}")
    return ds


# ═══════════════════════════════════════════════════════════════════════════════
# Path resolution
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve a path relative to *base* (defaults to PROJECT_ROOT).

    Absolute paths are returned as-is.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    base = Path(base) if base else _PROJECT_ROOT
    return (base / path).resolve()


if __name__ == "__main__":
    # Quick sanity check
    info = log_system_info()
    print(f"Git commit: {get_git_commit_hash()}")
    print(f"Project root: {_PROJECT_ROOT}")
