import enum
import json
from pathlib import Path
import joblib
import torch
from torch.utils.data import Dataset

DATASET_DIR = Path("dataset_cache/")
DATASET_DIR.mkdir(exist_ok=True, parents=True)


class Parts(enum.Enum):
    TRAIN = "train"
    VALID = "valid"


class Cord19Dataset(Dataset):
    def __init__(self, part: Parts):
        print("loading " + f'cord19-{part}')
        self.part = part.value
        with open(DATASET_DIR / Path("dataset_meta.json"), 'r') as json_file:
            self.meta = json.load(json_file)
        self.files_loaded = {}

    def __len__(self):
        return self.meta[f"{self.part}_size"]

    def __getitem__(self, index):
        file, true_index = self.get_file_from_index(index)
        if file not in self.files_loaded:
            print(f"Loading dataset file {file}...")
            self.files_loaded[file] = joblib.load(DATASET_DIR / Path(file))
        return (
            torch.tensor(self.files_loaded[file].input_ids[index], dtype=torch.int64),
            torch.tensor(self.files_loaded[file].target_ids[index], dtype=torch.int64)
        )

    def get_file_from_index(self, index):
        count = 0
        for file in self.meta[self.part]:
            count += self.meta[file]
            if count > index:
                return file, count - index
