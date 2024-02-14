# Tiny Quick Draw

## Introduction
A minimalist version of the the Google ["Quick, Draw !"](https://quickdraw.withgoogle.com/). Test your drawing skills against an AI that guesses what you are sketching !

<div align="center">
    <img src="https://private-user-images.githubusercontent.com/56489418/304731948-9fa76195-818e-451e-b2e0-809ba11aefe4.gif?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDc5MTM5MzgsIm5iZiI6MTcwNzkxMzYzOCwicGF0aCI6Ii81NjQ4OTQxOC8zMDQ3MzE5NDgtOWZhNzYxOTUtODE4ZS00NTFlLWIyZTAtODA5YmExMWFlZmU0LmdpZj9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAyMTQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMjE0VDEyMjcxOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTc1YWIyNDEzNDRlNTQyMmRlMjAzZmI3MzNjMjc4Yjk2YzRlZTlmZjEzNWVmMmNlYzMxZjNmODU5MGIwODMwMjUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.bavwKySAtNRQIsMBBKsZLBhySFT1aF18gpHymYtRwAY" width="320" height="320"/>
</div>

## Architecture
For this project, I chose reimplement a ResNet, a popular architecture known for its proficiency in complex image recognition tasks.
This ResNet was trained from scratch to see where I can go.

This model is implemented with TensorFlow and Keras.

<div align="center">
	<img src="https://private-user-images.githubusercontent.com/56489418/304732205-bb495042-72f6-4c48-b20c-ad4aac911945.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDc5MTM5MzgsIm5iZiI6MTcwNzkxMzYzOCwicGF0aCI6Ii81NjQ4OTQxOC8zMDQ3MzIyMDUtYmI0OTUwNDItNzJmNi00YzQ4LWIyMGMtYWQ0YWFjOTExOTQ1LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAyMTQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMjE0VDEyMjcxOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTJkZmNiYjZlZGMzNGY3ZmY1ODE0NjVhNjhmOGQ0ZDljNDY2NjEzNjU4YWMzOWZlNjNiYzgxNGZmNzE2ODU4NDUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.8wlgwXDLv4TS3RfmQ-G4F4dCV_joeqen9R8ULwb7f5c" alt="ResNet Bloc"/>
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
make run MODEL_PATH=path/to/model.h5 CONFIG_PATH=path/to/the/corresponding/config.yaml
```
Here, `path/to/model.keras` should be replaced with the path to your trained model file, and `path/to/the/corresponding/config.yaml` with the path to the configuration file used for training this model.

## Track metrics with [Weight & Biases](https://wandb.ai/)

To track advanced metrics and visualize results such as a confusion matrix, specify wandb_parameters in your configuration file. This allows integration with Weights & Biases for performance tracking and visualization.

Example configuration snippet:

yaml
Copy code
wandb_parameters:
  project_name: project name
  username: username
After specifying these parameters, the training process will log metrics to your Weights & Biases dashboard, where you can view detailed performance charts and confusion matrices.

<div align="center">
    <!-- Placeholder for Metrics Curve Image -->
    <img src="https://private-user-images.githubusercontent.com/56489418/304731746-b2115205-e5c2-4273-aca6-9468abe0870c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDc5MTM5MzgsIm5iZiI6MTcwNzkxMzYzOCwicGF0aCI6Ii81NjQ4OTQxOC8zMDQ3MzE3NDYtYjIxMTUyMDUtZTVjMi00MjczLWFjYTYtOTQ2OGFiZTA4NzBjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAyMTQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMjE0VDEyMjcxOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTJmZDg2Y2E0YTQ2Njg0ZmM3NWU2MWEyMjAyZmJiNjdlNjYyMjdlNWZkZDMwYzNlZGZjY2RiZDgwY2I5NWFiZDgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.XfRkaOCmBv4_VOpltdHhv0sshLRCE7SyQkljNNR1zDc" alt="Metrics Curve" width="480" height="320"/>
    <!-- Placeholder for Confusion Matrix Image -->
    <img src="https://private-user-images.githubusercontent.com/56489418/304731852-592e87cc-97b2-4809-ae5b-380e2355538f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDc5MTM5MzgsIm5iZiI6MTcwNzkxMzYzOCwicGF0aCI6Ii81NjQ4OTQxOC8zMDQ3MzE4NTItNTkyZTg3Y2MtOTdiMi00ODA5LWFlNWItMzgwZTIzNTU1MzhmLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAyMTQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMjE0VDEyMjcxOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWIzODk4MGEyMGUzZDhlZjE2ODI0M2UxNDg1ZGZiNzJmYThiZDg5NTE2MDI5ZmVmNzFmNjg5NTQzNDg1YTM1MWUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.PI0-cqt0_hRYGggqmtkmfoaVRlXwHg6Nu4uZug_o-Qg" alt="Confusion Matrix" width="480" height="400"/>
</div>
