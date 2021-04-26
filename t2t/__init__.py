import math
from functools import partial
from dataclasses import dataclass
from typing import Callable, Union

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning_spells as pls
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer, MT5ForConditionalGeneration, MT5Tokenizer
)


def masked_cross_entropy_loss(outputs, targets):
    # print(targets["ids"].shape, outputs.shape)
    targets, mask = targets["ids"], targets["mask"]
    loss = torch.sum(
        mask.view(-1) * F.cross_entropy(
            outputs.view(-1, outputs.size(2)),
            targets.view(-1),
            reduction="none"
        )
    ) / mask.sum()
    return loss


def single_token_cross_entropy_loss(outputs, targets):
    target_ids = targets["ids"]
    # print(outputs[:, :, 333])
    loss = F.cross_entropy(
        outputs.view(-1, outputs.size(2)),
        target_ids.view(-1)
    )
    return loss


def optimize_sequence(ids, pad, max_len):
    # Pad to the minimum multiple of 8 to utilize tensor cores
    batch_max_len = np.max([x.size(0) for x in ids])
    if batch_max_len > 8:
        max_length = math.ceil(
            min(max_len, batch_max_len) / 8.
        ) * 8
    else:
        max_length = batch_max_len
    padded_ids = ids[0].new_zeros((len(ids), max_length)) + pad
    mask = ids[0].new_zeros((len(ids), max_length))
    for i, example in enumerate(ids):
        example = example[:max_len]
        padded_ids[i, :len(example)] = example
        mask[i, :len(example)] = 1
    return padded_ids, mask


def collate_batch(batch, max_len, pad=0, decode_start_token=0, is_classifier=False):
    """Batch preparation.

    Truncate the sequence to reduce wastes.
    """
    source_ids, target_ids = zip(*batch)
    source_ids, src_mask = optimize_sequence(source_ids, pad, max_len)
    if not is_classifier:
        target_ids, target_mask = optimize_sequence(target_ids, pad, max_len)
        shifted_target_ids = target_ids.new_zeros(target_ids.shape)
        shifted_target_ids[..., 1:] = target_ids[..., :-1].clone()
        shifted_target_ids[..., 0] = decode_start_token
        return (
            {
                "input_ids": source_ids, "attention_mask": src_mask,
                "decoder_input_ids": shifted_target_ids
            },
            {"ids": target_ids, "mask": target_mask}
        )
    # is classifier
    target_ids = torch.stack(target_ids)
    shifted_target_ids = target_ids.new_zeros(target_ids.shape[0], 1) + decode_start_token
    return (
        {
            "input_ids": source_ids, "attention_mask": src_mask,
            "decoder_input_ids": shifted_target_ids
        },
        {"ids": target_ids}
    )


class SortSampler(Sampler):
    def __init__(self, data_source, key):
        self.data_source, self.key = data_source, key

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        return iter(sorted(range(len(self.data_source)), key=self.key, reverse=True))


class SortishSampler(Sampler):
    """Returns an iterator that traverses the the data in randomly ordered batches that are approximately the same size.
    The max key size batch is always returned in the first call because of pytorch cuda memory allocation sequencing.
    Without that max key returned first multiple buffers may be allocated when the first created isn't large enough
    to hold the next in the sequence.

    Taken from fastai library.
    """

    def __init__(self, data_source, key, bs, chunk_size=50):
        self.data_source, self.key, self.bs = data_source, key, bs
        self.chunk_size = 50

    def __len__(self): return len(self.data_source)

    def __iter__(self):
        while True:
            idxs = np.random.permutation(len(self.data_source))
            sz = self.bs * self.chunk_size
            ck_idx = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
            sort_idx = np.concatenate(
                [sorted(s, key=self.key, reverse=True) for s in ck_idx])
            sz = self.bs
            ck_idx = [sort_idx[i:i+sz]for i in range(0, len(sort_idx), sz)]
            # find the chunk with the largest key,
            max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])
            # then make sure it goes first.
            if len(ck_idx[max_ck]) != self.bs:
                continue
            ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]
            sort_idx = np.concatenate(np.random.permutation([
                np.random.permutation(chunk.reshape(self.bs, -1)).reshape(-1)
                for chunk in ck_idx[1:-1]
            ]))
            sort_idx = np.concatenate((ck_idx[0], sort_idx, ck_idx[-1]))
            break
        return iter(sort_idx)


@dataclass
class BaseConfig:
    base_t5_model: str
    batch_size: int
    fp16: bool
    learning_rate: float
    weight_decay: float
    epochs: int
    max_len: int
    loss_fn: Callable
    num_gpus: int = 1
    grad_accu: int = 1
    tpu_cores: int = 0


class T5BaseModel(pl.LightningModule):
    def __init__(
            self, config: BaseConfig, model: Union[T5ForConditionalGeneration, MT5ForConditionalGeneration],
            tokenizer: Union[T5Tokenizer, MT5Tokenizer], is_classifier: bool = False, **kwargs):
        super().__init__()
        self.config = config
        # placeholder for pylint
        self.train_dataset: Dataset = Dataset()
        self.valid_dataset: Dataset = Dataset()
        # the actual stuffs
        self.model = model
        self.tokenizer = tokenizer
        self.collate_fn = partial(
            collate_batch, pad=self.model.config.decoder_start_token_id,
            decode_start_token=self.model.config.pad_token_id,
            max_len=self.config.max_len, is_classifier=is_classifier
        )
        self.metrics = [
            ("acc", pl.metrics.Accuracy(compute_on_step=False))
        ]
        self.train_loss_tracker = pls.utils.EMATracker(alpha=0.02)

    def forward(self, input_tensors):
        return self.model(**input_tensors)[0]

    def train_dataloader(self):
        if self.config.tpu_cores:
            sampler = None
        else:
            sampler = SortishSampler(
                self.train_dataset,
                key=lambda x: len(self.train_dataset[x][0]),
                bs=self.config.batch_size
            )
        return DataLoader(
            self.train_dataset, num_workers=4 if self.config.tpu_cores == 0 else 1, shuffle=False, drop_last=True,
            batch_size=self.config.batch_size, collate_fn=self.collate_fn, sampler=sampler)

    def get_progress_bar_dict(self):
        # don't show the experiment version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, num_workers=4 if self.config.tpu_cores == 0 else 1, shuffle=False, drop_last=False,
            batch_size=self.config.batch_size*2, collate_fn=self.collate_fn)

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch[0])
        loss = self.config.loss_fn(
            logits,
            batch[1]
        )
        preds = torch.argmax(logits, dim=-1)[:, :batch[1]["ids"].size(1)]
        return {
            'loss': loss,
            'preds': preds,
            'target': batch[1]
        }

    def validation_step_end(self, outputs):
        self.log('val_loss', outputs['loss'].mean())
        for name, metric in self.metrics:
            metric(
                outputs['preds'].view(-1).cpu(),
                outputs['target']['ids'].view(-1).cpu()
            )
            self.log("val_" + name, metric)

    def _should_log(self, flag):
        if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
            if isinstance(flag, list):
                return flag[0]
            return flag
        return False

    def training_step_end(self, outputs):
        loss = outputs["loss"].mean()
        self.train_loss_tracker.update(loss.detach())
        if self._should_log(outputs["log"]):
            self.logger.log_metrics({
                "train_loss": self.train_loss_tracker.value
            }, step=self.global_step)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.config.loss_fn(
            self.forward(batch[0]),
            batch[1]
        )
        return {"loss": loss, "log": batch_idx % self.trainer.accumulate_grad_batches == 0}

    def configure_optimizers(self):
        optimizer = pls.optimizers.RAdam(
            self.model.parameters(), lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        steps_per_epochs = math.floor(
            len(self.train_dataset) / self.config.batch_size / self.config.grad_accu  # / self.num_gpus # dpp mode
        )
        print("Steps per epochs:", steps_per_epochs)
        n_steps = steps_per_epochs * self.config.epochs
        lr_durations = [
            int(n_steps*0.05),
            int(np.ceil(n_steps*0.95)) + 1
        ]
        break_points = [0] + list(np.cumsum(lr_durations))[:-1]
        scheduler = {
            'scheduler': pls.lr_schedulers.MultiStageScheduler(
                [
                    pls.lr_schedulers.LinearLR(optimizer, 0.01, lr_durations[0]),
                    CosineAnnealingLR(optimizer, lr_durations[1])
                ],
                start_at_epochs=break_points
            ),
            'interval': 'step',
            'frequency': 1,
            'strict': True,
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
