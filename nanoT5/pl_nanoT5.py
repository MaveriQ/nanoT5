from pytorch_lightning import LightningModule, LightningDataModule
import hydra
import logging
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from omegaconf import open_dict

from utils import (
    setup_basics,
    load_dataset_splits,
    process_dataset,
    DataCollatorForT5MLM,
    get_lr_scheduler,
    get_optimizer,
    get_tokenizer,
    get_model,
    compute_input_and_target_lengths,
    get_config,
)

class T5LightningDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = get_tokenizer(args)
        self.config = get_config(args)
        self.data_collator = DataCollatorForT5MLM(
            tokenizer=self.tokenizer,
            noise_density=args.data.mlm_probability,
            mean_noise_span_length=args.data.mean_noise_span_length,
            input_length=args.data.input_length,
            target_length=args.data.target_length,
            pad_token_id=self.config.pad_token_id,
        )

    def prepare_data(self):
        dataset = load_from_disk('/storage/ukp/work/jabbar/git/nanoT5/c4_small')
        dataset = dataset.remove_columns(
            ['timestamp', 'url']
        )
        dataset_splits = {
            'train': dataset['train'],
            'test': dataset['validation'],
        }
        # dataset_splits = load_dataset_splits(self.args)
        dataset_splits['train'] = dataset_splits['train'].select(range(self.args.data.max_train_samples))
        self.dataset = process_dataset(dataset_splits=dataset_splits, args=self.args, tokenizer=self.tokenizer)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.args.optim.batch_size,
            num_workers=self.args.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def validation_dataloader(self):
        return DataLoader(
            self.dataset['test'],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.args.optim.batch_size*2,
            num_workers=self.args.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    
class T5LightningModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = get_config(args)
        self.model = get_model(args, self.config)
        setup_basics(args)

    def forward(self, x):
        return self.model(**x)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        correct = (outputs.logits.argmax(-1) == batch["labels"]).sum().item()
        accuracy = correct / batch["labels"].numel()
        return {"val_loss": loss, "val_acc": accuracy}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        self.log("val_acc", avg_acc)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.model, self.args)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.args, None)
        return self.optimizer, self.lr_scheduler
    
def compute_lengths(args):

    before_mask_input_length, target_length = compute_input_and_target_lengths(
                inputs_length=args.data.input_length,
                noise_density=args.data.mlm_probability,
                mean_noise_span_length=args.data.mean_noise_span_length,
            )
    with open_dict(args):
        args.data.before_mask_input_length = before_mask_input_length
        args.data.target_length = target_length
    
@hydra.main(config_path="configs", config_name="default", version_base='1.1')
def main(args):
    args.data.streaming = False
    args.device = "cpu"
    compute_lengths(args)
    dm = T5LightningDataModule(args)
    dm.prepare_data()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    print(batch)
    model = T5LightningModule(args)
    out = model.training_step(batch,0)
    print(out)

if __name__ == "__main__":
    main()