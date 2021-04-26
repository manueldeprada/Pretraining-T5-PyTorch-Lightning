import enum
import json
from pathlib import Path
import joblib
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

DATASET_DIR = Path("dataset_cache/")
DATASET_DIR.mkdir(exist_ok=True, parents=True)


class Parts(enum.Enum):
    TRAIN = "train"
    VALID = "valid"


class Cord19Dataset(Dataset):
    def __init__(self, part: Parts):
        self.part = part.value
        print(f"Loading {self.part} dataset into memory...")
        with open(DATASET_DIR / Path("dataset_meta.json"), 'r') as json_file:
            self.meta = json.load(json_file)
        self.files_loaded = {}
        for file in tqdm(self.meta[self.part], ncols=100):
            self.files_loaded[file] = joblib.load(DATASET_DIR / Path(file))

    def __len__(self):
        return self.meta[f"{self.part}_size"]

    def __getitem__(self, index):
        file, true_index = self.get_file_from_index(index)
        # if file not in self.files_loaded:
        #     print(f"Loading dataset file {file}...")
        #     self.files_loaded[file] = joblib.load(DATASET_DIR / Path(file))
        return (
            torch.tensor(self.files_loaded[file][0][true_index], dtype=torch.int64),  # inputs in files_loaded[file][0]
            torch.tensor(self.files_loaded[file][1][true_index], dtype=torch.int64)  # targets in files_loaded[file][1]
        )

    def get_file_from_index(self, index):
        count = 0
        for file in self.meta[self.part]:
            count += self.meta[file]
            if index < count:
                return file, index - count + self.meta[file]

