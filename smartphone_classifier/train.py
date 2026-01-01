import logging
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from smartphone_classifier.dataset import SmartphoneDataset
from smartphone_classifier.model import build_model
from smartphone_classifier.download_data import build_dataframe_from_remote
from smartphone_classifier.transforms import get_train_transforms, get_test_transforms
from smartphone_classifier.train_loop import train_epoch, validate
import matplotlib.pyplot as plt


class DummyMLflow:
    def set_tracking_uri(self, uri):
        pass

    def set_experiment(self, name):
        pass

    def start_run(self):
        class DummyRun:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return DummyRun()

    def log_param(self, k, v):
        pass

    def log_params(self, d):
        pass

    def log_metric(self, k, v, step=None):
        pass

    def log_artifact(self, path):
        pass


mlflow = DummyMLflow()


logger = logging.getLogger(__name__)


def get_git_commit_id() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def train(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "batch_size": cfg.train.batch_size,
                "device": cfg.train.device,
                "num_classes": cfg.model.num_classes,
                "test_size": cfg.data.test_size,
            }
        )

        mlflow.log_param("git_commit_id", get_git_commit_id())

        df = build_dataframe_from_remote(remote_name="myremote", data_path="models")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        if cfg.train.get("num_samples", None) is not None:
            df = df.head(cfg.train.num_samples)
        print(df.head())
        print(len(df))

        train_df, val_df = train_test_split(
            df,
            test_size=cfg.data.test_size,
            random_state=cfg.data.random_seed,
            stratify=df["models_label"],
        )

        train_ds = SmartphoneDataset(train_df, get_train_transforms())
        img, label = train_ds[0]
        print("Image shape:", img.shape)
        print("Label:", label)

        val_ds = SmartphoneDataset(val_df, get_test_transforms())
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.train.batch_size,
        )

        device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

        model = build_model(
            cfg.model.num_classes,
            cfg.model.pretrained,
        ).to(device)

        criterion = nn.CrossEntropyLoss()

        for param in model.parameters():
            param.requires_grad = False

        best_val_acc = 0.0
        global_step = 0

        logs_dir = Path(cfg.train.output_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        train_losses, train_accs, val_accs = [], [], []

        for stage, stage_cfg in cfg.train.stages.items():
            logger.info(f"Stage {stage}")

            params = []
            for layer in stage_cfg.layers:
                if layer == "head":
                    for p in model.heads.head.parameters():
                        p.requires_grad = True
                    params.append(
                        {"params": model.heads.head.parameters(), "lr": stage_cfg.lr}
                    )
                else:
                    for p in model.encoder.layers[layer].parameters():
                        p.requires_grad = True
                    params.append(
                        {
                            "params": model.encoder.layers[layer].parameters(),
                            "lr": stage_cfg.lr,
                        }
                    )

            optimizer = optim.Adam(params, weight_decay=stage_cfg.wd)

            for epoch in range(stage_cfg.epochs):
                train_metrics = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                )
                val_metrics = validate(
                    model,
                    val_loader,
                    criterion,
                    device,
                )

                mlflow.log_metric("train_loss", train_metrics["loss"], step=global_step)
                mlflow.log_metric(
                    "train_accuracy", train_metrics["accuracy"], step=global_step
                )
                mlflow.log_metric(
                    "val_accuracy", val_metrics["accuracy"], step=global_step
                )

                train_losses.append(train_metrics["loss"])
                train_accs.append(train_metrics["accuracy"])
                val_accs.append(val_metrics["accuracy"])

                logger.info(
                    f"Stage {stage} | Epoch {epoch+1}/{stage_cfg.epochs} | "
                    f"Train loss: {train_metrics['loss']:.4f}, "
                    f"Train acc: {train_metrics['accuracy']:.4f}, "
                    f"Val acc: {val_metrics['accuracy']:.4f}"
                )

                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
                    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)
                    model_path = Path(cfg.train.output_dir) / "best.pth"
                    torch.save(model.state_dict(), model_path)
                    mlflow.log_artifact(model_path)

                global_step += 1

        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train Loss per Epoch")
        plt.legend()
        plt.savefig(logs_dir / "train_loss.png")
        plt.close()

        plt.figure()
        plt.plot(range(1, len(train_accs) + 1), train_accs, label="Train Acc")
        plt.plot(range(1, len(val_accs) + 1), val_accs, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy per Epoch")
        plt.legend()
        plt.savefig(logs_dir / "accuracy.png")
        plt.close()
