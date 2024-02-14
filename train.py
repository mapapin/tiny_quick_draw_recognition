import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from argparse import ArgumentParser  # noqa: E402
from pathlib import Path  # noqa: E402

from common import (  # noqa: E402
    generate_run_name,
    generate_model,
    get_callbacks,
    get_config,
    close_wandb_session,
    wandb_confusion_matrix,
)
from data import generate_data, get_train_val_datasets  # noqa: E402


def training(config) -> None:
    generate_run_name(config)
    generate_data(config)

    train_gen, val_gen, test_gen = get_train_val_datasets(config)

    model = generate_model(config)

    callbacks = get_callbacks(config)

    model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        epochs=config.epochs,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(test_gen, steps=test_gen.samples // test_gen.batch_size)
    print(f"Test loss: {test_loss} - Test accuracy: {test_accuracy}")

    if config.wandb_parameters:
        wandb_confusion_matrix(model, test_gen)
        close_wandb_session()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_path", nargs="?", default="config.yaml", type=Path)

    args = parser.parse_args()
    config = get_config(args.config_path)

    training(config)
