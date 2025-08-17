from typing import Any, Dict, Tuple

from tensorflow.keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from qrennd.configs import Config


def get_filename(params: Dict[str, Any]) -> str:
    if params["save_best_only"]:
        filename = "weights.keras"
        return filename

    filename = f"weights-{{epoch}}-{{{params['monitor']}}}.keras"
    return filename


def get_callbacks(config: Config) -> Tuple[Callback]:
    params = config.train["callbacks"]

    checkpoint_filename = get_filename(params["checkpoint"])
    model_checkpoint = ModelCheckpoint(
        filepath=str(config.checkpoint_dir / checkpoint_filename),
        **params["checkpoint"],
    )

    early_stop = EarlyStopping(
        **params["early_stop"],
    )

    csv_logs = CSVLogger(
        filename=str(config.log_dir / "training.log"),
        **params["csv_log"],
    )

    """
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=24,
        verbose=1,
        mode="auto",
    )

    return model_checkpoint, early_stop, csv_logs, reduce_lr
    """
    return model_checkpoint, early_stop, csv_logs
