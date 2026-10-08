"""
Distributed Hyperparameter Optimization Worker for Optuna and Ray.
Manages asynchronous trial lifecycle, device isolation, and exception handling.
"""

from typing import Dict, Any, Callable
import traceback
import optuna


def define_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample continuous and categorical hyperparameters for sequential models."""
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "tcn_channels": trial.suggest_categorical("tcn_channels", ["32,64", "64,128"]),
        "tfm_nhead": trial.suggest_categorical("tfm_nhead", [2, 4]),
        "tfm_layers": trial.suggest_int("tfm_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.05, 0.3),
    }


def objective_worker(
    trial: optuna.Trial,
    gpu_queue,
    training_params: dict,
    data_pipeline_func: Callable,
    model_factory_func: Callable,
    trainer_factory_func: Callable,
    eval_metric: str = "sharpe",
    seed: int = 42
) -> float:
    """
    Distributed worker function.
    Safely acquires GPU index from thread-safe queue, runs isolated trial,
    and guarantees resource release back to queue upon completion or error.
    """
    gpu_id = -1
    try:
        # Retrieve device token from thread-safe queue
        if gpu_queue is not None:
            gpu_id = gpu_queue.get(timeout=60)
            device = f"cuda:{gpu_id}"
        else:
            device = "cpu"

        # Hyperparameter sampling
        hparams = define_search_space(trial)

        # Build pipeline and models with sampled parameters
        train_loader, val_loader = data_pipeline_func(batch_size=training_params.get("batch_size", 64))
        model = model_factory_func(hparams=hparams, device=device)
        trainer = trainer_factory_func(model=model, train_loader=train_loader, val_loader=val_loader, hparams=hparams, device=device)

        # Trial execution with pruning checks
        best_score = -float("inf")
        epochs = training_params.get("epochs", 5)
        for ep in range(epochs):
            train_loss = trainer.train_epoch()
            eval_metrics = trainer.evaluate()
            
            # Primary validation signal
            score = eval_metrics.get("hit_rate", 0.0) - (eval_metrics.get("val_loss", 1.0) * 0.1)
            best_score = max(best_score, score)

            # Report intermediate step to Optuna pruner
            trial.report(score, step=ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_score

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as exc:
        traceback.print_exc()
        trial.set_user_attr("worker_error", traceback.format_exc())
        raise exc
    finally:
        # Guarantee device index is always restored
        if gpu_id != -1 and gpu_queue is not None:
            gpu_queue.put(gpu_id)
