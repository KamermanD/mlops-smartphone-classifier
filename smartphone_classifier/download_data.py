from pathlib import Path
import pandas as pd
from dvc.api import DVCFileSystem


def build_dataframe_from_remote(
    remote_name="myremote",
    data_path="models",
) -> pd.DataFrame:
    fs = DVCFileSystem(repo=".", remote=remote_name)

    data = {
        "img": [],
        "models": [],
    }

    classes = [e for e in fs.ls(data_path) if e["type"] == "directory"]

    for cls in classes:
        cls_path = cls["name"]
        cls_name = Path(cls_path).name

        files = [f for f in fs.ls(cls_path) if f["type"] == "file"]

        for f in files:
            data["img"].append(f["name"])
            data["models"].append(cls_name)

    df = pd.DataFrame(data)

    model_to_label = {m: i for i, m in enumerate(df["models"].unique())}
    df["models_label"] = df["models"].map(model_to_label)

    return df
