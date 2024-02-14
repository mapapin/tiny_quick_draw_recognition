import numpy as np
from PIL import Image
from quickdraw import QuickDrawDataGroup
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import shutil

from schemas import Config


def generate_class_images(config, name, num_train, num_val, num_test):
    base_directory = config.data.folder
    image_size = config.image_size

    # Calculer le nombre total de dessins
    total_drawings = num_train + num_val + num_test

    # Créer les dossiers s'ils n'existent pas
    for subset in ["train", "validation", "test"]:
        (base_directory / subset / name).mkdir(parents=True, exist_ok=True)

    images = QuickDrawDataGroup(name, max_drawings=total_drawings, recognized=True, print_messages=False)

    for i, img in enumerate(images.drawings):
        if i < num_train:
            subset = "train"
        elif i < num_train + num_val:
            subset = "validation"
        else:
            subset = "test"

        filename = base_directory / subset / name / f"{img.key_id}.png"
        img = img.get_image(stroke_width=3).resize(image_size).convert("L")

        img = np.array(img) / 255.0
        threshold = 0.9
        img = np.where(img <= threshold, 0, 1)

        Image.fromarray(img.astype(np.uint8)).save(filename)


def generate_data(config: Config) -> None:
    if not config.data.generate:
        return

    if (
        config.data.generate
        and config.data.folder.exists()
        and any(elem.is_dir() for elem in config.data.folder.iterdir())
    ):
        msg = (
            f"The folder at {config.data.folder} already exists and is not empty."
            "Do you still want to regenerate the data? [Y/n] "
        )
        res = input(msg)
        while res not in {"Y", "y", "N", "n"}:
            res = input(msg)

        if res in {"N", "n"}:
            return
        else:
            shutil.rmtree(config.data.folder)

    for label in tqdm(config.data.classes):
        generate_class_images(
            config, label, config.data.train_samples, config.data.validation_samples, config.data.test_samples
        )


def get_train_val_datasets(config: Config) -> tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
    train_datagen = ImageDataGenerator()
    validation_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        config.data.folder / "train",
        target_size=config.image_size,
        batch_size=config.data.batch_size,
        class_mode="sparse",
        color_mode="grayscale",
        shuffle=True,
    )

    validation_generator = validation_datagen.flow_from_directory(
        config.data.folder / "validation",
        target_size=config.image_size,
        batch_size=config.data.batch_size,
        class_mode="sparse",
        color_mode="grayscale",
        shuffle=False,
    )

    test_generator = test_datagen.flow_from_directory(
        config.data.folder / "test",
        target_size=config.image_size,
        batch_size=config.data.batch_size,
        class_mode="sparse",
        color_mode="grayscale",
        shuffle=False,
    )

    return train_generator, validation_generator, test_generator
