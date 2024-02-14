import datetime
from pathlib import Path
from typing import Any

import yaml
from keras.callbacks import EarlyStopping, ModelCheckpoint

from schemas import Config
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from keras.models import Model
from keras.preprocessing.image import DirectoryIterator

from resnet import ResNet34
from keras.optimizers import Adam  # noqa: E402


def generate_run_name(config: Config) -> None:
    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.loss_name}_lr:{config.learning_rate}_bs:{config.data.batch_size}_{datetime_str}"
    config.run_name = run_name


def generate_model(config: Config, weights_path: Path | None = None) -> Model:
    model = ResNet34(config.image_size + (1,), len(config.data.classes))

    optimizer = Adam(learning_rate=config.learning_rate)

    if weights_path is not None:
        model.load_weights(weights_path)

    model.compile(optimizer=optimizer, loss=config.loss, metrics=["accuracy"])

    return model


def get_callbacks(config: Config) -> list[Any]:
    callbacks = []
    if config.wandb_parameters is not None:
        try:
            import wandb
            from wandb.keras import WandbCallback

            wandb.ensure_configured()
            if wandb.run is None:
                wandb.init(
                    project=config.wandb_parameters.project_name,
                    entity=config.wandb_parameters.username,
                    name=config.run_name,
                )
                wandb_config = {
                    "learning_rate": config.learning_rate,
                    "epochs": config.epochs,
                    "batch_size": config.data.batch_size,
                    "loss_name": config.loss_name,
                    "image_size": config.image_size,
                    "train_samples": config.data.train_samples,
                    "validation_samples": config.data.validation_samples,
                    "test_samples": config.data.test_samples,
                    "classes": config.data.classes,
                    "n_classes": len(config.data.classes),
                }
                if config.loss_parameters is not None:
                    wandb_config.update(config.loss_parameters)

                wandb.config.update(wandb_config, allow_val_change=True)

                callbacks.append(
                    WandbCallback(log_batch_fequency=config.wandb_parameters.log_batch_fequency, save_weights_only=True)
                )

        except ImportError as e:
            raise ImportError(
                "You chose to use wandb by filling wandb_parameters in the configuration file but it's not installed."
                " Please install it or run without these configuration options."
            ) from e

    best_model_path = f"./models/best_model-{config.run_name}.keras"
    callbacks.append(
        ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True, mode="min", save_weights_only=True)
    )
    callbacks.append(EarlyStopping(monitor="val_loss", patience=3))

    return callbacks


def get_config(config_path: Path) -> Config:
    with open(config_path, "r") as config_file:
        config_data = yaml.safe_load(config_file)

    config = Config(**config_data)

    return config


def wandb_confusion_matrix(model: Model, test_generator: DirectoryIterator) -> None:
    import wandb

    y_pred = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    y_true = test_generator.classes

    if len(y_pred_classes) != len(y_true):
        raise ValueError("Le nombre de prédictions ne correspond pas au nombre de vrais labels.")

    class_names = list(test_generator.class_indices.keys())

    plt.figure(figsize=(15, 15))

    font_size = 6
    plt.rcParams.update({"font.size": font_size})

    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    conf_matrix_disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    conf_matrix_disp.plot(include_values=False)

    fig = plt.gcf()
    ax = plt.gca()

    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

    plt.tight_layout()

    wandb.log({"conf_matrix_plot": wandb.Image(fig)})

    plt.close(fig)


def close_wandb_session() -> None:
    import wandb

    wandb.finish()
