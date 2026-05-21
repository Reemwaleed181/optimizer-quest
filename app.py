import os
import threading
import time
from flask import Flask, render_template, jsonify, request
from train import train_model

app = Flask(__name__)

training_lock = threading.Lock()
training_state = {
    "status": "Ready",
    "current_epoch": 0,
    "total_epochs": 0,
    "progress": 0,
    "elapsed_time_seconds": 0,
    "optimizer": None,
    "dataset": None,
    "model": None,
    "loss_function": None,
    "learning_rate": None,
    "batch_size": None,
    "metrics": {},
    "result": None,
    "error": None
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "message": "Backend is running"
    })


def update_training_state(**kwargs):
    with training_lock:
        training_state.update(kwargs)


def get_training_state():
    with training_lock:
        return dict(training_state)


def is_hosted_demo():
    return (
        os.environ.get("RENDER") == "true"
        or bool(os.environ.get("RAILWAY_ENVIRONMENT"))
    )


def apply_hosted_limits(epochs, batch_size):
    if not is_hosted_demo():
        return epochs, batch_size

    return min(epochs, 1), min(batch_size, 32)


def run_training_job(optimizer_name, lr, epochs, batch_size, dataset_name, model_name, loss_name):
    start_time = time.time()

    def on_epoch(payload):
        update_training_state(
            current_epoch=payload["current_epoch"],
            total_epochs=payload["total_epochs"],
            progress=payload["progress"],
            elapsed_time_seconds=payload["elapsed_time_seconds"],
            metrics={
                "train_loss": payload["train_loss"],
                "train_accuracy": payload["train_accuracy"],
                "validation_loss": payload["validation_loss"],
                "validation_accuracy": payload["validation_accuracy"],
                "epoch_time_seconds": payload["epoch_time_seconds"]
            }
        )

    try:
        result = train_model(
            optimizer_name=optimizer_name,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            dataset_name=dataset_name,
            model_name=model_name,
            loss_name=loss_name,
            progress_callback=on_epoch
        )

        update_training_state(
            status="Completed",
            current_epoch=result["epochs"],
            total_epochs=result["epochs"],
            progress=100,
            elapsed_time_seconds=result["training_time_seconds"],
            metrics={
                "train_loss": result["final_train_loss"],
                "train_accuracy": result["final_train_accuracy"],
                "validation_loss": result["final_validation_loss"],
                "validation_accuracy": result["final_validation_accuracy"],
                "test_loss": result["final_test_loss"],
                "test_accuracy": result["final_test_accuracy"],
                "epoch_time_seconds": (
                    result["epoch_times"][-1] if result["epoch_times"] else None
                )
            },
            result=result,
            error=None
        )

    except Exception as exc:
        update_training_state(
            status="Ready",
            elapsed_time_seconds=round(time.time() - start_time, 2),
            error=str(exc)
        )


@app.route("/api/train", methods=["POST"])
def api_train():
    try:
        data = request.get_json() or {}

        optimizer_name = data.get("optimizer", "adam")
        dataset_name = data.get("dataset", "mnist")
        model_name = data.get("model", "simplecnn")
        loss_name = data.get("loss_function", "crossentropy")
        lr = float(data.get("learning_rate", 0.001))
        epochs = int(data.get("epochs", 2))
        batch_size = int(data.get("batch_size", 64))
        epochs, batch_size = apply_hosted_limits(epochs, batch_size)

        with training_lock:
            if training_state["status"] == "Training":
                return jsonify({
                    "status": "error",
                    "message": "Training is already running"
                }), 409

            training_state.update({
                "status": "Training",
                "current_epoch": 0,
                "total_epochs": epochs,
                "progress": 0,
                "elapsed_time_seconds": 0,
                "optimizer": optimizer_name,
                "dataset": dataset_name,
                "model": model_name,
                "loss_function": loss_name,
                "learning_rate": lr,
                "batch_size": batch_size,
                "metrics": {},
                "result": None,
                "error": None
            })

        worker = threading.Thread(
            target=run_training_job,
            args=(optimizer_name, lr, epochs, batch_size, dataset_name, model_name, loss_name),
            daemon=True
        )
        worker.start()

        return jsonify({
            "status": "success",
            "message": "Training started",
            "state": get_training_state()
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


@app.route("/api/train/status")
def api_train_status():
    state = get_training_state()
    return jsonify({
        "status": "success",
        "state": state
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True, use_reloader=False)
