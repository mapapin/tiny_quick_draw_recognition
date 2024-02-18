# Tiny Quick Draw

## Introduction
A minimalist version of the the Google ["Quick, Draw !"](https://quickdraw.withgoogle.com/). Test your drawing skills against an AI that guesses what you are sketching !

<div align="center">
    <img src="assets/tiny_quick_draw_demo.gif" width="320" height="320"/>
</div>

## Architecture
For this project, I chose reimplement a ResNet, a popular architecture known for its proficiency in complex image recognition tasks.
This ResNet was trained from scratch to see where I can go.

This model is implemented with TensorFlow and Keras.

<div align="center">
	<img src="assets/resnet_bloc.ppm" alt="ResNet Bloc"/>
	<figcaption><i>ResNet bloc</i></figcaption>
</div>

## Data
The data for this project are sourced directly from the [quickdraw](https://pypi.org/project/quickdraw/) package. This package provides an accessible way to fetch a diverse array of sketches from the extensive Google Quick Draw collection.

## Interface
The user interface is powered by Pygame, chosen for its simplicity and effectiveness in redering graphics.
It provides a simple canvas for users to draw their sketches.

If you want to reset the canvas, press <kbd>r</kbd>.

## Setup
```
git clone https://github.com/mapapin/tiny_quick_draw.git
cd tiny_quick_draw
make install
source venv/bin/activate
make run
```

## Clean env
```
make clean
```

## Retrain the Model
To retrain the model with your custom parameters, follow these steps:

### 1. Prepare your configuration file in the YAML format.
You should use the following structure for the configuration file:
```yaml
# Example Configuration File

# Mandatory. Define the size of the input images (width, height).
image_size: [28, 28]

# Mandatory. Specify the type of loss function. Choose between 'focal' and 'cross_entropy'.
loss_name: "cross_entropy"

# Optional. Parameters specific to the chosen loss function. Can be omitted or defined with necessary key-value pairs.
loss_parameters:
  # For example, if using 'focal' loss, you might define parameters like 'gamma' and 'alpha'.
  # gamma: 2.0
  # alpha: 0.25

# Mandatory. Number of training epochs.
epochs: 15

# Mandatory. Learning rate for the optimizer.
learning_rate: 0.0001

# Optional. Weights & Biases (wandb) configuration.
wandb_parameters:
  project_name: "wandb project name" # Mandatory if using wandb.
  username: "username"               # Mandatory if using wandb.
  log_batch_frequency: 10            # Optional. Frequency of logging batches.

# Mandatory. Data configuration.
data:
  folder: "dataset"               # Mandatory. Path to the dataset folder.
  generate: True                  # Mandatory. Whether to generate data if not present.
  train_samples: 3000             # Mandatory. Number of training samples.
  validation_samples: 500         # Mandatory. Number of validation samples.
  test_samples: 500               # Mandatory. Number of test samples.
  batch_size: 64                  # Mandatory. Size of each data batch.
  classes:                        # Mandatory. List of classes for the model to recognize.
    - airplane
    - apple
    # ... other classes ...
    - tree
```


### 2. Start the training process using the command:

```
make train
```

This command will use the default configuration file.

### 3. If you wish to use a custom configuration file, execute the following command:

```
make train CONFIG_PATH=path/to/config.yaml
```

Replace `path/to/config.yaml` with the actual path to your custom YAML configuration file.

After retraining the model with your desired parameters, you may want to run the interface with the newly trained model. Execute the command below to do so:

```
make run MODEL_PATH=path/to/model.keras CONFIG_PATH=path/to/the/corresponding/config.yaml
```
Here, `path/to/model.keras` should be replaced with the path to your trained model file, and `path/to/the/corresponding/config.yaml` with the path to the configuration file used for training this model.

## Track metrics with [Weight & Biases](https://wandb.ai/)

To track advanced metrics and visualize results such as a confusion matrix, specify wandb_parameters in your configuration file. This allows integration with Weights & Biases for performance tracking and visualization.
> Please note that you must also have wandb installed and be logged in before executing the `make train` command to have access to metrics.
Configuration snippet:

```yaml
wandb_parameters:
  project_name: project name
  username: username
```

After specifying these parameters, the training process will log metrics to your Weights & Biases dashboard, where you can view detailed performance charts and confusion matrices.

<div align="center">
    <!-- Placeholder for Metrics Curve Image -->
    <img src="assets/wandb_metrics.png" alt="Metrics Curve" width="480" height="320"/>
    <!-- Placeholder for Confusion Matrix Image -->
    <img src="assets/wandb_confusion_matrix.png" alt="Confusion Matrix" width="480" height="400"/>
</div>
