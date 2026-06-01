import math

import torch


def _isnan(v):
    return v is None or math.isnan(v)


def _log_metric(key, value, step):
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_metric(key, value, step=step)
    except ImportError:
        pass


def train_loop(
    model,
    train_loader,
    val_loader,
    train_fn,
    eval_fn,
    optimizer,
    max_epochs,
    patience,
    checkpoint_path,
):
    """Generic training loop with early stopping."""
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_fn(model, train_loader, optimizer)
        metrics = eval_fn(model, val_loader)
        val_loss = metrics["loss"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        _log_metric("train_loss", train_loss, epoch)
        for k, v in metrics.items():
            _log_metric(k, v, epoch)

        parts = " | ".join(f"{k}: {v:.6f}" for k, v in metrics.items())
        print(f"Epoch {epoch:3d} | train_loss: {train_loss:.6f} | {parts}")

        if val_loss is None or _isnan(train_loss) or _isnan(val_loss):
            print(f"NaN detected at epoch {epoch}, stopping early")
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch} epochs (patience={patience})")
                break

    return {
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
