"""
NeuralGCM India Fine-Tuning -- Entry Point
==========================================
Usage:
  python neuralgcm_finetune/run_finetune.py --verify-data
  python neuralgcm_finetune/run_finetune.py --download-ckpt
  python neuralgcm_finetune/run_finetune.py --phase a
  python neuralgcm_finetune/run_finetune.py --phase b
  python neuralgcm_finetune/run_finetune.py --evaluate
  python neuralgcm_finetune/run_finetune.py --all
"""

import os
import sys

os.environ["JAX_PLATFORMS"]                  = "cpu"
# XLA_FLAGS removed — xla_cpu_use_thunk_runtime is deprecated in JAX 0.9+
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]  = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import jax
_ = jax.devices()

import argparse
import yaml
import json
from pathlib import Path
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stdout, level="INFO",
    format="<green>{time:HH:mm:ss}</green> | "
           "<level>{level:<8}</level> | {message}")
Path("neuralgcm_finetune/logs").mkdir(parents=True, exist_ok=True)
logger.add(
    "neuralgcm_finetune/logs/training.log",
    level="DEBUG", rotation="50 MB")


def load_cfg() -> dict:
    """Load the fine-tuning configuration."""
    cfg_path = Path("neuralgcm_finetune/config_finetune.yaml")
    if not cfg_path.exists():
        logger.error(f"Config not found: {cfg_path}")
        sys.exit(1)
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_loader_and_sampler(config: dict):
    """Initialise data loader and sampler."""
    from neuralgcm_finetune.data.loader import ERA5GRIBLoader
    from neuralgcm_finetune.data.sampler import IndiaERA5Sampler

    loader = ERA5GRIBLoader(config["data"]["grib_dir"], config)
    loader.load_all(use_cache=True)
    sampler = IndiaERA5Sampler(loader, config)
    return loader, sampler


def cmd_verify(args):
    """Verify all 7 GRIB files are readable and merge correctly."""
    config = load_cfg()
    grib_d = Path(config["data"]["grib_dir"])

    print(f"\n{'=' * 60}")
    print("  Verifying GRIB dataset")
    print(f"  Directory: {grib_d.absolute()}")
    print(f"{'=' * 60}")

    all_files = (
        config["data"]["single_level_files"] +
        config["data"]["pressure_level_files"])
    ok = True
    total_size = 0
    for fname in all_files:
        fp = grib_d / fname
        if fp.exists():
            size_mb = fp.stat().st_size / 1e6
            total_size += size_mb
            print(f"  OK  {fname:<30} {size_mb:>8.1f} MB")
        else:
            print(f"  XX  {fname:<30} NOT FOUND")
            ok = False

    if not ok:
        print(
            "\n  ERROR: Missing files. Check that all 7 GRIB "
            "files are in the 'Fine Tuning/' folder.")
        sys.exit(1)

    print(f"\n  Total size: {total_size:.0f} MB "
          f"({total_size / 1000:.1f} GB)")

    # Quick read test
    print("\n  Loading and merging (first run builds cache)...")
    loader, sampler = get_loader_and_sampler(config)
    times = loader.get_available_times()

    print(f"\n  Total time steps: {len(times)}")
    if times:
        print(f"  First: {times[0].strftime('%Y-%m-%d %H:%M')}")
        print(f"  Last:  {times[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"  Train: {len(sampler.train_times)} steps")
    print(f"  Test:  {len(sampler.test_times)} steps "
          "(Dec 2024)")

    # Show dataset variables
    ds = loader.merged_ds
    if ds is not None:
        print(f"\n  Dimensions: {dict(ds.sizes)}")
        vars_list = list(ds.data_vars)
        print(f"  Variables ({len(vars_list)}):")
        for i in range(0, len(vars_list), 5):
            chunk = vars_list[i:i + 5]
            print(f"    {', '.join(chunk)}")
        if "level" in ds.coords:
            levs = sorted(ds.level.values.tolist())
            print(f"  Pressure levels: {levs}")

    print(f"\n  Dataset verified successfully.")
    print(f"{'=' * 60}\n")
    loader.verify_log_surface_pressure()


def cmd_download(args):
    """Download base NeuralGCM checkpoint."""
    config = load_cfg()
    import shutil

    existing = Path(config["model"].get(
        "existing_checkpoint_cache",
        "cache/v1_deterministic_2_8_deg.pkl"))
    target = Path(config["model"]["base_checkpoint_local"])
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        logger.info(f"Checkpoint already exists: {target}")
        return

    if existing.exists():
        logger.info(f"Copying from cache: {existing} -> {target}")
        shutil.copy2(str(existing), str(target))
        logger.success(f"Checkpoint ready: {target}")
        return

    # Download from GCS
    from neuralgcm_weather.model.checkpoint import (
        load_checkpoint)
    logger.info("Downloading base checkpoint from GCS...")
    load_checkpoint(
        model_name=config["model"]["base_checkpoint_gcs"],
        local_cache_dir=str(target.parent),
    )
    logger.success("Base checkpoint ready.")


def cmd_phase(args):
    """Run a training phase."""
    config = load_cfg()
    loader, sampler = get_loader_and_sampler(config)

    from neuralgcm_finetune.training.trainer import IndiaFineTuner
    trainer = IndiaFineTuner(config, loader, sampler)
    trainer.setup()
    history = trainer.train_phase(args.phase)

    # Save training history
    h_path = (f"neuralgcm_finetune/logs/"
              f"history_phase_{args.phase}.json")
    with open(h_path, "w") as f:
        json.dump(history, f, default=str, indent=2)
    logger.success(f"History saved: {h_path}")
    plot_training_curves(args.phase)

    # After phase B, save final checkpoint
    if args.phase == "b":
        out = trainer.save_finetuned_checkpoint()
        if out:
            print(f"\n  Fine-tuned checkpoint: {out}")
            print(
                "  To use in forecasts, load with:\n"
                f"    neuralgcm.PressureLevelModel."
                f"from_checkpoint('{out}')")

def plot_training_curves(phase: str):
    """Plot train loss and val loss from saved history JSON."""
    import json, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    h_path = (f"neuralgcm_finetune/logs/"
              f"history_phase_{phase}.json")
    if not Path(h_path).exists():
        return

    with open(h_path) as f:
        history = json.load(f)

    steps = history.get("step", [])
    losses = history.get("loss", [])
    val_data = history.get("val_loss", [])

    DARK  = "#0d1117"; PANEL = "#161b22"; BORDER = "#30363d"
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK)
    ax.set_facecolor(PANEL)

    if steps and losses:
        ax.plot(steps, losses, color="#58A6FF", lw=1.5,
                alpha=0.7, label="Train loss")

        # Moving average
        if len(losses) > 20:
            import numpy as np
            ma = np.convolve(losses,
                             np.ones(20)/20, mode="valid")
            ax.plot(steps[19:], ma, color="#58A6FF",
                    lw=2.5, label="Train loss (MA-20)")

    if val_data:
        val_steps = [v["step"] for v in val_data]
        val_losses = [v["val"] for v in val_data]
        ax.plot(val_steps, val_losses, color="#3FB950",
                lw=2.5, marker="o", ms=5,
                label="Validation loss")

    ax.set_title(
        f"Phase {phase.upper()} Training Curves — "
        f"India ERA5 2024",
        color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Step", color="white", fontsize=10)
    ax.set_ylabel("Loss", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.grid(True, color=BORDER, ls="--", alpha=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.legend(fontsize=9, facecolor=PANEL,
              labelcolor="white", edgecolor=BORDER)

    out = Path("neuralgcm_finetune/results")
    out.mkdir(parents=True, exist_ok=True)
    outpath = out / f"training_curves_phase_{phase}.png"
    plt.savefig(outpath, dpi=150,
                bbox_inches="tight", facecolor=DARK)
    plt.close()
    from loguru import logger
    logger.success(f"Training curve saved: {outpath}")

def cmd_test_gradient(args):
    """
    Test that gradients flow correctly through the fine-tuning setup.
    Run this before starting Phase A to catch wiring bugs early.
    """
    config  = load_cfg()
    loader, sampler = get_loader_and_sampler(config)

    from neuralgcm_finetune.training.trainer import IndiaFineTuner
    trainer = IndiaFineTuner(config, loader, sampler)
    trainer.setup()

    logger.info("Testing gradient computation (Phase A wiring)...")
    trainable, frozen = trainer._split_params("a")

    # Sample one tiny batch
    batch = sampler.sample_training_batch(
        batch_size=1,
        rollout_steps=1,
        pressure_vars=config["model"]["input_vars"],
        surface_vars=config["model"]["surface_vars"],
        timestep_hours=6,
    )
    if not batch:
        logger.error("No training batch available — check dataset")
        return

    import jax
    import jax.numpy as jnp

    rng = jax.random.key(0)

    def loss_fn(tp):
        init_ds, targets = batch[0]
        l, _ = trainer._compute_step_loss(
            tp, frozen, init_ds, targets, 1, "a", rng)
        return l

    logger.info("  Computing value_and_grad...")
    try:
        val, grads = jax.value_and_grad(loss_fn)(trainable)
        logger.success(f"  Loss value: {float(val):.6f}")

        # Check that gradients are non-zero and non-NaN
        grad_leaves = jax.tree.leaves(grads)
        n_nonzero = sum(
            1 for g in grad_leaves
            if jnp.any(g != 0))
        n_nan = sum(
            1 for g in grad_leaves
            if jnp.any(jnp.isnan(g)))
        n_total = len(grad_leaves)

        logger.info(
            f"  Gradient stats: "
            f"{n_nonzero}/{n_total} arrays non-zero | "
            f"{n_nan} NaN arrays")

        if n_nan > 0:
            logger.error(
                f"  {n_nan} gradient arrays contain NaN! "
                "Check loss.py for division by zero in "
                "area_weights or spectral_mse.")
        if n_nonzero == 0:
            logger.error(
                "  ALL gradients are zero! "
                "The trainable params are not in the "
                "computation graph. Fix _compute_step_loss "
                "to use trainable params explicitly.")
        if n_nonzero > 0 and n_nan == 0:
            logger.success(
                "  Gradient test PASSED. "
                "Fine-tuning is correctly wired.")
    except Exception as e:
        logger.error(f"  Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_evaluate(args):
    """Run evaluation comparing base vs fine-tuned model."""
    config = load_cfg()
    loader, sampler = get_loader_and_sampler(config)

    from neuralgcm_finetune.evaluation.evaluator import (
        IndiaEvaluator)
    ev = IndiaEvaluator(config, loader, sampler.test_times)
    df = ev.run()

    out = Path("neuralgcm_finetune/results")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "evaluation_report.csv", index=False)
    logger.success(
        "CSV: neuralgcm_finetune/results/evaluation_report.csv")

    ev.plot(df)
    ev.print_summary(df)


def cmd_all(args):
    """Run all phases and evaluation."""
    # Phase A
    logger.info("Starting Phase A (decoder fine-tuning)...")
    args.phase = "a"
    cmd_phase(args)

    # Phase B
    logger.info("Starting Phase B (physics fine-tuning)...")
    args.phase = "b"
    cmd_phase(args)

    # Evaluation
    logger.info("Starting evaluation...")
    cmd_evaluate(args)


def main():
    p = argparse.ArgumentParser(
        description="NeuralGCM India 2024 Fine-Tuning")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--verify-data", action="store_true",
        help="Verify all 7 GRIB files load correctly")
    g.add_argument(
        "--download-ckpt", action="store_true",
        help="Download/cache base NeuralGCM checkpoint")
    g.add_argument(
        "--phase", choices=["a", "b"],
        help="Run training phase (a=decoder, b=physics)")
    g.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate base vs fine-tuned model")
    g.add_argument(
        "--test-gradient", action="store_true",
        help="Verify gradient flow before training")
    g.add_argument(
        "--all", action="store_true",
        help="Run all phases then evaluate")
    args = p.parse_args()

    if args.verify_data:
        cmd_verify(args)
    elif args.download_ckpt:
        cmd_download(args)
    elif args.phase:
        cmd_phase(args)
    elif args.evaluate:
        cmd_evaluate(args)
    elif args.test_gradient:
        cmd_test_gradient(args)
    elif args.all:
        cmd_all(args)


if __name__ == "__main__":
    main()
