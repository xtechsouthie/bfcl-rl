"""
Phase 1 — SFT Training for Qwen3-1.7B Tool-Use.

Uses:
  - Unsloth for QLoRA kernels
  - TRL SFTTrainer
  - RapidFire AI for hyperparameter sweeps over LoRA r/alpha
  - Liger kernels for optimized attention
  - WandB for experiment tracking

Run:
    python training/sft_train.py
    python training/sft_train.py --config configs/sft_config.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Allow running from project root
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils import (
    GracefulShutdown,
    init_wandb,
    load_config,
    log_system_info,
    resolve_path,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Apply Liger kernels BEFORE model instantiation
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2
    apply_liger_kernel_to_qwen2()
    print("✅ Liger kernels applied for Qwen2/3")
except ImportError:
    print("⚠️  liger-kernel not installed, proceeding without optimized kernels")
except Exception as e:
    print(f"⚠️  Failed to apply Liger kernels: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SFT Config dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SFTRunConfig:
    """All SFT training configuration, loaded from YAML."""
    # Model
    model_name: str = "Qwen/Qwen3-1.7B"
    model_cache_dir: str = "models/qwen3-1.7b"

    # Dataset
    dataset_name: str = "Salesforce/xlam-function-calling-60k"
    dataset_cache_dir: str = "data/sft_processed"
    train_size: int = 30000
    eval_size: int = 1500
    seed: int = 42

    # QLoRA
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"

    # LoRA
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"
    lora_r_values: List[int] = field(default_factory=lambda: [16, 32])
    lora_alpha_values: List[int] = field(default_factory=lambda: [16, 32])

    # Training
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 2
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True
    max_seq_length: int = 2048
    packing: bool = True
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    report_to: str = "wandb"

    # RapidFire AI
    experiment_name: str = "sft-qwen3-1.7b-tool-use"
    num_chunks: int = 4
    early_stop_threshold: float = 1.5

    # Output
    output_dir: str = "checkpoints/sft_runs"
    best_checkpoint_dir: str = "checkpoints/sft_best"

    # Liger
    use_liger_kernel: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SFTRunConfig":
        """Create config from dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_sft_dataset(cfg: SFTRunConfig) -> tuple:
    """Load the processed SFT dataset.

    Returns (train_dataset, eval_dataset).
    """
    from datasets import load_from_disk

    cache_dir = resolve_path(cfg.dataset_cache_dir)
    if not cache_dir.exists():
        print(f"⚠️  Processed dataset not found at {cache_dir}")
        print("   Run: python data/prepare_sft_data.py first!")
        sys.exit(1)

    ds = load_from_disk(str(cache_dir))
    train_ds = ds["train"]
    eval_ds = ds["eval"]

    print(f"Loaded SFT dataset: {len(train_ds)} train, {len(eval_ds)} eval")
    return train_ds, eval_ds


def prepare_messages_for_training(
    dataset: Any,
    tokenizer: Any,
) -> Any:
    """Convert stored JSON messages to the format expected by SFTTrainer.

    The dataset has a 'messages' column storing JSON strings of chat messages.
    We parse them back into lists of dicts.
    """
    def _parse_messages(example: Dict[str, Any]) -> Dict[str, Any]:
        messages = json.loads(example["messages"])
        return {"messages": messages}

    return dataset.map(
        _parse_messages,
        num_proc=4,
        desc="Preparing messages",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Model + Tokenizer loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(
    cfg: SFTRunConfig,
    lora_r: int,
    lora_alpha: int,
) -> tuple:
    """Load Qwen3-1.7B with QLoRA configuration and LoRA adapter.

    Returns (model, tokenizer).
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_path = resolve_path(cfg.model_cache_dir)

    # QLoRA config
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    compute_dtype = dtype_map.get(cfg.bnb_4bit_compute_dtype, torch.bfloat16)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
    )

    # Download / load model
    print(f"Loading model: {cfg.model_name}")
    try:
        from huggingface_hub import snapshot_download
        if not model_path.exists():
            print(f"Downloading model to {model_path}…")
            snapshot_download(
                repo_id=cfg.model_name,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
            )
    except Exception as e:
        print(f"Model download note: {e}")

    # Try loading from local cache first, then from HF
    load_path = str(model_path) if model_path.exists() else cfg.model_name

    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        load_path,
        trust_remote_code=True,
    )

    # Unsloth or certain configs set the eos_token to a placeholder like "<EOS_TOKEN>"
    # TRL 0.24+ strictly checks if eos_token is in the actual vocabulary.
    # We must properly register whatever eos_token and pad_token are set to.
    
    # <|im_end|> is natively in the Qwen vocabulary. Point eos/pad to it to bypass TRL checks.
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = "<|im_end|>"

    # Prepare for QLoRA
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        task_type=cfg.lora_task_type,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_single_run(
    cfg: SFTRunConfig,
    lora_r: int,
    lora_alpha: int,
    train_dataset: Any,
    eval_dataset: Any,
    run_name: str,
) -> Dict[str, Any]:
    """Run a single SFT training with the given LoRA config.

    Returns a dict with final metrics.
    """
    from trl import SFTConfig, SFTTrainer
    import wandb

    output_dir = resolve_path(cfg.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Init WandB ────────────────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    wandb_run = init_wandb(
        experiment_name=run_name,
        config={
            "model_name": cfg.model_name,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.per_device_train_batch_size,
            "grad_accum": cfg.gradient_accumulation_steps,
            "epochs": cfg.num_train_epochs,
            "max_seq_length": cfg.max_seq_length,
            "packing": cfg.packing,
        },
        extra_id=f"r{lora_r}-a{lora_alpha}-{timestamp}",
        tags=["sft", f"r{lora_r}", f"a{lora_alpha}"],
    )

    # ── Log system info ──────────────────────────────────────────────────
    sys_info = log_system_info()
    wandb.config.update({"system_info": sys_info})

    # ── Load model ────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg, lora_r, lora_alpha)

    # ── Prepare dataset messages ──────────────────────────────────────────
    train_ds = prepare_messages_for_training(train_dataset, tokenizer)
    eval_ds = prepare_messages_for_training(eval_dataset, tokenizer)

    # ── SFT Config ────────────────────────────────────────────────────────
    # Compute warmup_steps from ratio (warmup_ratio deprecated in transformers v5.2)
    effective_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    total_steps = (cfg.train_size // effective_batch) * cfg.num_train_epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        warmup_steps=warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=cfg.bf16,
        tf32=cfg.tf32,
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_length=cfg.max_seq_length,
        packing=cfg.packing,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        report_to=cfg.report_to,
        run_name=run_name,
        seed=cfg.seed,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # ── Train with graceful shutdown ──────────────────────────────────────
    def emergency_save() -> None:
        print(f"Emergency save to {output_dir}/emergency_checkpoint")
        trainer.save_model(str(output_dir / "emergency_checkpoint"))
        wandb.finish(quiet=True)

    results: Dict[str, Any] = {}
    with GracefulShutdown(emergency_save):
        train_result = trainer.train()
        results = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "run_name": run_name,
        }

        # Eval
        eval_results = trainer.evaluate()
        results["eval_loss"] = eval_results.get("eval_loss", float("inf"))
        results["eval_runtime"] = eval_results.get("eval_runtime", 0)

        # Log final metrics
        wandb.log({
            "final/train_loss": results["train_loss"],
            "final/eval_loss": results["eval_loss"],
        })

    # ── Save model ────────────────────────────────────────────────────────
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"✅ Saved model to {final_path}")

    # ── Log sample outputs ────────────────────────────────────────────────
    try:
        _log_sample_outputs(model, tokenizer, eval_ds, wandb)
    except Exception as e:
        print(f"⚠️  Failed to log sample outputs: {e}")

    wandb.finish()

    # Cleanup GPU memory
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def _log_sample_outputs(
    model: Any,
    tokenizer: Any,
    eval_ds: Any,
    wandb: Any,
    num_samples: int = 5,
) -> None:
    """Generate and log sample outputs to WandB as a Table."""
    import wandb as wb

    model.eval()
    table = wb.Table(columns=["input", "expected", "generated"])

    for i in range(min(num_samples, len(eval_ds))):
        messages = eval_ds[i]["messages"]
        # Use only system + user messages as input
        input_msgs = [m for m in messages if m["role"] in ("system", "user")]
        expected = [m for m in messages if m["role"] == "assistant"]
        expected_text = expected[0]["content"] if expected else "N/A"

        prompt = tokenizer.apply_chat_template(
            input_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        table.add_data(
            prompt[:500],
            expected_text[:500],
            generated[:500],
        )

    wb.log({"eval/sample_outputs": table})


# ═══════════════════════════════════════════════════════════════════════════════
# RapidFire AI Grid Search
# ═══════════════════════════════════════════════════════════════════════════════

def run_rapidfire_sweep(cfg: SFTRunConfig) -> None:
    """Run grid search over LoRA r/alpha using RapidFire AI.

    Sweeps over r in {16, 32} with alpha = r.
    Stops underperforming runs at chunk 2 if loss > 1.5x best.
    """
    try:
        from rapidfireai import Experiment
        from rapidfireai.search import RFGridSearch
        from rapidfireai.automl import RFSFTConfig
        _use_rapidfire = True
    except ImportError:
        print("Warning: rapidfireai not installed -- falling back to manual grid search")
        _use_rapidfire = False

    train_ds, eval_ds = load_sft_dataset(cfg)

    if _use_rapidfire:
        _run_rapidfire_sweep(cfg, train_ds, eval_ds)
    else:
        _run_manual_sweep(cfg, train_ds, eval_ds)


def _run_rapidfire_sweep(
    cfg: SFTRunConfig,
    train_ds: Any,
    eval_ds: Any,
) -> None:
    """Run sweep using the RapidFire AI Experiment / RFGridSearch API."""
    import shutil
    from rapidfireai import Experiment
    from rapidfireai.search import RFGridSearch
    from rapidfireai.automl import RFSFTConfig

    # ── 1. Pre-process dataset messages (parse JSON strings -> dicts) ------
    print("Pre-processing dataset messages for RF sweep...")

    def _parse_messages(example: Dict[str, Any]) -> Dict[str, Any]:
        return {"messages": json.loads(example["messages"])}

    train_ready = train_ds.map(_parse_messages, num_proc=4, desc="Prep train")
    eval_ready = eval_ds.map(_parse_messages, num_proc=4, desc="Prep eval")

    # ── 2. create_model_fn: called per config by RapidFire AI --------------
    # RF resolves the grid and passes each config as a plain dict.
    def create_model_fn(config: Dict[str, Any]) -> Any:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        compute_dtype = dtype_map.get(cfg.bnb_4bit_compute_dtype, torch.bfloat16)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        )

        model_path = resolve_path(cfg.model_cache_dir)
        load_path = str(model_path) if model_path.exists() else cfg.model_name

        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        # RF passes the resolved grid values as a dict
        chosen_r     = config.get("r", 32)
        chosen_alpha = config.get("lora_alpha", chosen_r)

        lora_config = LoraConfig(
            r=chosen_r,
            lora_alpha=chosen_alpha,
            target_modules=cfg.lora_target_modules,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.lora_bias,
            task_type=cfg.lora_task_type,
        )
        return get_peft_model(model, lora_config)

    # ── 3. Build RFGridSearch with a dict (not kwargs) ---------------------
    # alpha always equals r; zip ensures we sweep paired values.
    paired_r     = [r for r, a in zip(cfg.lora_r_values, cfg.lora_alpha_values) if r == a]
    paired_alpha = [a for r, a in zip(cfg.lora_r_values, cfg.lora_alpha_values) if r == a]

    output_dir = resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute warmup_steps from ratio (warmup_ratio deprecated in transformers v5.2)
    effective_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    total_steps = (cfg.train_size // effective_batch) * cfg.num_train_epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    grid = RFGridSearch({
        # LoRA params to sweep — RF generates all combos from lists
        "r": paired_r,
        "lora_alpha": paired_alpha,
        # SFT training config (static across all runs)
        "sft_config": RFSFTConfig(
            output_dir=str(output_dir),
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            num_train_epochs=cfg.num_train_epochs,
            warmup_steps=warmup_steps,
            lr_scheduler_type=cfg.lr_scheduler_type,
            bf16=cfg.bf16,
            tf32=cfg.tf32,
            gradient_checkpointing=cfg.gradient_checkpointing,
            max_length=cfg.max_seq_length,
            packing=cfg.packing,
            logging_steps=cfg.logging_steps,
            eval_strategy="steps",
            eval_steps=cfg.eval_steps,
            save_strategy="steps",
            save_steps=cfg.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=cfg.report_to,
            seed=cfg.seed,
            remove_unused_columns=False,
        ),
    })

    # ── 4. Create Experiment and run_fit -----------------------------------
    experiment = Experiment(
        experiment_name=cfg.experiment_name,
        mode="fit",
    )

    print(f"\nLaunching RapidFire AI sweep: {cfg.experiment_name}")
    print(f"  Configs : r={paired_r}  (alpha = r)")
    print(f"  Chunks  : {cfg.num_chunks}")
    print(f"  Early-stop: {cfg.early_stop_threshold}x best loss at chunk 2\n")

    experiment.run_fit(
        config_group=grid,
        create_model_fn=create_model_fn,
        train_dataset=train_ready,
        eval_dataset=eval_ready,
        num_chunks=cfg.num_chunks,
    )

    # ── 5. Copy best checkpoint to checkpoints/sft_best/ ------------------
    best_dir = resolve_path(cfg.best_checkpoint_dir)
    try:
        best_ckpt_path = experiment.best_checkpoint_path()
        if best_ckpt_path and Path(best_ckpt_path).exists():
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(best_ckpt_path, str(best_dir))
            best_config = experiment.best_config()
            print(f"Best checkpoint (r={best_config.get('r', '?')}) "
                  f"saved to {best_dir}")
        else:
            print(f"RF did not return a best checkpoint path; "
                  f"check {output_dir} manually.")
    except AttributeError:
        print("experiment.best_config() not available in this RF version; "
              "please pick the best run from the RF dashboard and copy it to "
              f"{best_dir} manually.")


def _run_manual_sweep(
    cfg: SFTRunConfig,
    train_ds: Any,
    eval_ds: Any,
) -> None:
    """Fallback manual grid search if RapidFire AI is not available."""
    # Only sweep over r=alpha pairs
    param_combos = [
        (r, a) for r, a in zip(cfg.lora_r_values, cfg.lora_alpha_values)
        if r == a
    ]

    best_loss = float("inf")
    best_run: Optional[Dict[str, Any]] = None
    all_results: List[Dict[str, Any]] = []

    for i, (r, alpha) in enumerate(param_combos):
        run_name = f"{cfg.experiment_name}-r{r}-a{alpha}"

        print(f"\n{'='*60}")
        print(f"  SWEEP RUN {i+1}/{len(param_combos)}: r={r}, alpha={alpha}")
        print(f"{'='*60}")

        results = train_single_run(
            cfg=cfg,
            lora_r=r,
            lora_alpha=alpha,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            run_name=run_name,
        )
        all_results.append(results)

        eval_loss = results.get("eval_loss", float("inf"))
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_run = results

        # Early stopping
        if i >= 1 and eval_loss > cfg.early_stop_threshold * best_loss:
            print(f"⚠️  Run r={r} eval_loss={eval_loss:.4f} exceeds "
                  f"{cfg.early_stop_threshold}x best ({best_loss:.4f}), stopping sweep")
            break

    _save_best_model(cfg, best_run, all_results)


def _save_best_model(
    cfg: SFTRunConfig,
    best_run: Optional[Dict[str, Any]],
    all_results: List[Dict[str, Any]],
) -> None:
    """Copy the best run's model to the best checkpoint directory."""
    if best_run is None:
        print("❌ No successful runs")
        return

    best_dir = resolve_path(cfg.best_checkpoint_dir)
    run_dir = resolve_path(cfg.output_dir) / best_run["run_name"] / "final"

    print(f"\n{'='*60}")
    print(f"  BEST RUN: {best_run['run_name']}")
    print(f"  eval_loss = {best_run.get('eval_loss', 'N/A')}")
    print(f"  r={best_run['lora_r']}, alpha={best_run['lora_alpha']}")
    print(f"{'='*60}")

    # Copy best model
    if run_dir.exists():
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(str(run_dir), str(best_dir))
        print(f"✅ Saved best model to {best_dir}")
    else:
        print(f"⚠️  Best run directory not found: {run_dir}")

    # Log summary
    print("\n── Sweep Summary ──")
    for r in all_results:
        print(f"  {r['run_name']:40s}  eval_loss={r.get('eval_loss', 'N/A')}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: SFT Training")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_PROJECT_ROOT / "configs" / "sft_config.yaml"),
        help="Path to SFT config YAML",
    )
    parser.add_argument("--lora-r", type=int, default=None, help="Override LoRA r (single run)")
    parser.add_argument("--lora-alpha", type=int, default=None, help="Override LoRA alpha")
    args = parser.parse_args()

    config_dict = load_config(args.config)
    cfg = SFTRunConfig.from_dict(config_dict)

    if args.lora_r is not None:
        # Single run mode
        lora_r = args.lora_r
        lora_alpha = args.lora_alpha or lora_r  # alpha = r by default

        train_ds, eval_ds = load_sft_dataset(cfg)
        run_name = f"{cfg.experiment_name}-r{lora_r}-a{lora_alpha}"

        results = train_single_run(
            cfg=cfg,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            run_name=run_name,
        )
        print(f"\nResults: {json.dumps(results, indent=2, default=str)}")
    else:
        # Grid search mode
        run_rapidfire_sweep(cfg)


if __name__ == "__main__":
    main()
