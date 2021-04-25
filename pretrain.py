"""
Reference: https://github.com/ramsrigouthamg/Paraphrase-any-question-with-T5-Text-To-Text-Transfer-Transformer-
Next ref: https://github.com/ceshine/finetuning-t5/tree/master/paraphrase
"""
import enum
import os
from dataclasses import dataclass, asdict
import pytorch_lightning as pl
import pytorch_lightning_spells as pls
import typer
from transformers import T5ForConditionalGeneration, T5Tokenizer

from Cord19Dataset import Cord19Dataset, DATASET_DIR, Parts
from t2t import BaseConfig, T5BaseModel, masked_cross_entropy_loss


class Corpus(enum.Enum):
    CORD19 = "cord19"


@dataclass
class Config(BaseConfig):
    dataset: Corpus = Corpus.CORD19


class T5Model(T5BaseModel):
    def __init__(self, config: Config, **kwargs):
        model = T5ForConditionalGeneration.from_pretrained(config.base_t5_model)
        tokenizer = T5Tokenizer.from_pretrained(config.base_t5_model)
        super().__init__(config, model, tokenizer)
        self.config = config
        # log the config values
        self.save_hyperparameters(asdict(config))
        self.train_dataset = Cord19Dataset(Parts.TRAIN)
        print("Train dataset: ", len(self.train_dataset))
        self.valid_dataset = Cord19Dataset(Parts.VALID)
        print("Valid dataset: ", len(self.valid_dataset))


def main(
        t5_model: str = "t5-base", lr: float = 1e-4,  # 3^4
        epochs: int = 5, fp16: bool = False,
        dataset: Corpus = Corpus.CORD19, batch_size: int = 16,
        max_len: int = 64, grad_accu: int = 1,
        num_gpus: int = 1
):
    pl.seed_everything(int(os.environ.get("SEED", 738)))
    config = Config(
        base_t5_model=t5_model,
        learning_rate=lr,
        epochs=epochs,
        dataset=dataset,
        max_len=max_len,
        grad_accu=grad_accu,
        batch_size=batch_size,
        fp16=fp16,
        weight_decay=0,
        num_gpus=num_gpus,
        loss_fn=masked_cross_entropy_loss
    )

    pl_module = T5Model(config)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(DATASET_DIR / "model_checkpoints"),
            monitor='val_loss',
            mode="min",
            filename='{step:06d}-{val_loss:.4f}',
            save_top_k=1,
            save_last=False
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
    ]
    trainer = pl.Trainer(
        accelerator='dp' if num_gpus > 1 else None,
        # amp_backend="apex", amp_level='O2',
        precision=16 if config.fp16 else 32,
        gpus=config.num_gpus,
        val_check_interval=0.25,
        gradient_clip_val=10,
        max_epochs=epochs,
        # max_steps=steps,
        callbacks=callbacks,
        accumulate_grad_batches=grad_accu,
        # auto_scale_batch_size='power' if batch_size is None else None,
        logger=[
            pl.loggers.TensorBoardLogger(str(DATASET_DIR / "tb_logs"), name=""),
            pls.loggers.ScreenLogger(),
            # pl.loggers.WandbLogger(project="t5-paraphrase")
        ],
        log_every_n_steps=100
    )

    trainer.fit(pl_module)

    # pl_module.model.save_pretrained(CACHE_DIR / f"{config.base_t5_model}_last")
    # pl_module.tokenizer.save_pretrained(CACHE_DIR / f"{config.base_t5_model}_last")
    # print("Last model saved")

    assert isinstance(callbacks[0], pl.callbacks.ModelCheckpoint)
    print(callbacks[0].best_model_path)
    pl_module = T5Model.load_from_checkpoint(
        callbacks[0].best_model_path,
        config=config
    )
    pl_module.model.save_pretrained(DATASET_DIR / f"{config.base_t5_model}_best")
    pl_module.tokenizer.save_pretrained(DATASET_DIR / f"{config.base_t5_model}_best")
    print("Best model saved")


if __name__ == "__main__":
    typer.run(main)
