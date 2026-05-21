# Optimizer Quest

Optimizer Quest is an interactive Flask and PyTorch web application for comparing how different optimization algorithms behave during neural network training. The project turns a standard image-classification training loop into a visual experiment: users choose a dataset, model architecture, optimizer, learning rate, batch size, epoch count, and loss function, then watch training progress, metrics, charts, and run history update from the browser.

The goal is to make optimizer behavior easier to understand. Instead of only reading equations for SGD, Momentum, RMSprop, Adam, or AdamW, users can run the same controlled task with different optimizer choices and compare convergence speed, validation performance, test accuracy, and training stability.

## Features

- Interactive Flask web interface for configuring and launching training runs.
- PyTorch backend that trains real neural networks on image-classification datasets.
- Live training status with epoch progress, elapsed time, loss, and accuracy metrics.
- Optimizer comparison across SGD, Momentum, RMSprop, Adam, and AdamW.
- Automatic benchmark mode for running all optimizers under a fair comparison protocol.
- Dataset selection for MNIST, Fashion-MNIST, and KMNIST.
- Model selection for Tiny MLP, Simple CNN, and Deeper CNN architectures.
- Loss function selection for Cross-Entropy, Label-Smoothed Cross-Entropy, and MSE on class probabilities.
- Visual training arena and charts powered by the frontend JavaScript modules.
- Run history tracking with CSV export from the browser.
- Light and dark theme support.

## Project Purpose

Optimizer Quest was built as an educational machine learning demo. It helps students, researchers, and curious developers explore the practical effects of optimizer choice without needing to write a new training script for every experiment.

The app is useful for questions like:

- How quickly does Adam converge compared with plain SGD?
- Does Momentum improve early training stability?
- How does a high learning rate affect loss curves?
- Do deeper CNNs benefit more from adaptive optimizers?
- How do validation and test metrics change when using different objectives?

Because the experiments run through a real PyTorch training loop, the results reflect actual model behavior rather than a static animation.

## Tech Stack

- Backend: Flask
- Machine learning: PyTorch and torchvision
- Data handling: torchvision datasets and PyTorch DataLoader
- Frontend: HTML, CSS, and vanilla JavaScript
- Charts: Chart.js
- Runtime language: Python 3

## How It Works

The browser sends a training request to the Flask backend through `/api/train`. The backend starts a background training thread so the web server can keep responding while PyTorch trains the selected model.

During training, the PyTorch loop reports progress after each epoch. Flask stores the latest state in memory, and the frontend polls `/api/train/status` to update the UI. When training completes, the app stores final train, validation, and test metrics so the run can be compared against previous experiments.

The training pipeline includes:

1. Load the selected dataset.
2. Split the training data into train and validation sets.
3. Build the selected model architecture.
4. Create the selected loss function.
5. Create the selected optimizer.
6. Train for the chosen number of epochs.
7. Evaluate on the validation and test sets.
8. Return final metrics and epoch timing data to the frontend.

## Available Experiments

### Datasets

- MNIST: handwritten digit recognition with 10 digit classes.
- Fashion-MNIST: clothing image classification with 10 apparel classes.
- KMNIST: Japanese Kuzushiji character recognition with 10 classes.

Downloaded dataset files are stored locally in `data/`, which is ignored by Git.

### Model Architectures

- Tiny MLP: a lightweight fully connected network that flattens images before classification.
- Simple CNN: a balanced convolutional baseline with two convolution blocks.
- Deeper CNN: a larger convolutional model with batch normalization and dropout for stronger image-learning capacity.

### Optimizers

- SGD: standard stochastic gradient descent.
- Momentum: SGD with velocity accumulation.
- RMSprop: adaptive learning rate method based on moving averages of squared gradients.
- Adam: combines momentum-style updates with adaptive gradient scaling.
- AdamW: Adam with decoupled weight decay.

### Loss Functions

- Cross-Entropy: default classification loss for raw logits.
- Label-Smoothed Cross-Entropy: cross-entropy with softer labels for regularization.
- MSE on Class Probabilities: compares predicted probabilities against one-hot class targets.

## API Endpoints

### `GET /`

Renders the main Optimizer Quest interface.

### `GET /api/health`

Returns a simple backend health check.

### `POST /api/train`

Starts a training job in a background thread.

Example request body:

```json
{
  "optimizer": "adam",
  "dataset": "mnist",
  "model": "simplecnn",
  "loss_function": "crossentropy",
  "learning_rate": 0.001,
  "epochs": 2,
  "batch_size": 64
}
```

### `GET /api/train/status`

Returns the current training state, including status, progress, epoch count, elapsed time, selected configuration, metrics, final result, and any error message.

## Project Structure

```text
optimizer-quest/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        ├── app.js
        ├── arena.js
        ├── charts.js
        ├── datasets.js
        ├── nn.js
        └── optimizers.js
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

Start the Flask app:

```powershell
python app.py
```

Open the local URL printed by Flask, usually:

```text
http://127.0.0.1:5000
```

## Usage

1. Select an optimizer from the left panel.
2. Choose the dataset, model architecture, learning rate, batch size, epoch count, and loss function.
3. Click `Run Training` to start a single experiment.
4. Watch the progress bar, arena, and charts update as epochs complete.
5. Review final train, validation, and test metrics.
6. Use `Benchmark All Optimizers` to compare all supported optimizers under a consistent setup.
7. Export run history as CSV when you want to analyze results outside the app.

## Git Hygiene

The repository intentionally ignores generated and local-only files:

- `.venv/` and other virtual environment folders
- `data/` downloaded datasets
- Python cache files
- model checkpoints and training outputs
- local environment files

For a fresh checkout, install dependencies from `requirements.txt` instead of committing `.venv/`.

## Notes

- The first dataset run may take longer because torchvision downloads the dataset files.
- Training speed depends heavily on whether PyTorch can use CUDA or only CPU.
- Hosted free-tier deployments use the real PyTorch training pipeline with a smaller dataset subset and one epoch to stay within memory limits.
- Local runs can use the full dataset split and larger epoch counts.
- The Flask app stores training state in memory, so restarting the server clears the current run state and browser history.
- The app is designed for local experimentation and educational demos, not production model training.
