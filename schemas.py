from typing import Any, Literal

from focal_loss import SparseCategoricalFocalLoss
from pydantic import BaseModel, Field, validator
from quickdraw import QuickDrawData
from keras.losses import SparseCategoricalCrossentropy
from pathlib import Path


class WandbParameters(BaseModel):
    project_name: str = Field(min_length=1)
    username: str = Field(min_length=1)
    log_batch_fequency: int | None = None


class Data(BaseModel):
    folder: Path = Field(min_length=1)
    generate: bool = True
    train_samples: int
    validation_samples: int
    test_samples: int
    batch_size: int = Field(ge=1)
    classes: list[str] = Field(min_items=2)

    @validator("classes", pre=True, always=True)
    def verify_classes_names(cls, classes):
        qd = QuickDrawData()
        available_classes = set(qd.drawing_names)
        invalid_classes = set(classes) - available_classes

        if invalid_classes:
            raise ValueError(f"The following classes are not available: {', '.join(invalid_classes)}")

        return classes


class Config(BaseModel):
    run_name: str | None = None

    image_size: tuple[int, int]

    loss_name: Literal["focal", "cross_entropy"]
    loss_parameters: dict[str, Any] | None = None
    loss: Any = None

    @validator("loss", pre=True, always=True)
    def convert_loss_name_to_instance(cls, loss, values):
        loss_name = values.get("loss_name")
        loss_parameters = values.get("loss_parameters") or {}
        if loss_name == "focal":
            return SparseCategoricalFocalLoss(**loss_parameters)
        elif loss_name == "cross_entropy":
            return SparseCategoricalCrossentropy(**loss_parameters)
        else:
            raise ValueError(f"Loss '{loss_name}' is not recognized.")

    epochs: int = Field(ge=1)
    learning_rate: float = Field(ge=0)

    wandb_parameters: WandbParameters | None = None

    data: Data
