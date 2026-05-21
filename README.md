# Optimizer Quest

Optimizer Quest is a Flask web app for experimenting with neural network training choices. It lets you choose an optimizer, model, loss function, learning rate, epoch count, batch size, and dataset, then watch training progress and metrics update from the browser.

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

## Notes

- MNIST data is downloaded into `data/` when needed.
- Local virtual environments, downloaded datasets, caches, checkpoints, and training outputs are intentionally ignored by Git.
- For a fresh checkout, install dependencies from `requirements.txt` instead of committing `.venv/`.
