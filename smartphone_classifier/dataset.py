from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from dvc.api import DVCFileSystem


class SmartphoneDataset(Dataset):
    def __init__(
        self,
        df,
        transform=None,
        cache_dir="tmp",
        remote_name="myremote",
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.fs = DVCFileSystem(repo=".", remote=remote_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        remote_path = row["img"]
        label = row["models_label"]

        local_path = self.cache_dir / Path(remote_path).name

        if not local_path.exists():
            self.fs.get(remote_path, str(local_path))

        image = Image.open(local_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
