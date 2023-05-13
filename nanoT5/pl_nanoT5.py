from pytorch_lightning import LightningModule, LightningDataModule
import hydra
import logging

from utils import (
    setup_basics,
    train,
    predict,
    eval,
    get_lr_scheduler,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_dataloaders,
    get_config,
)

class T5LightningDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = get_tokenizer(args)
        self.config = get_config(args)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataloader, self.test_dataloader = get_dataloaders(self.tokenizer, self.config, self.args)

    def train_dataloader(self):
        return self.train_dataloader

    def test_dataloader(self):
        return self.test_dataloader
    
class T5LightningModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = get_config(args)
        self.model = get_model(args, self.config)
        self.logger = setup_basics(args)
        self.optimizer = get_optimizer(self.model, args)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, args, self.logger)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):

    def validation_step(self, batch, batch_idx):

    def configure_optimizers(self):
        return self.optimizer, self.lr_scheduler
    


@hydra.main(config_path="configs", config_name="default", version_base='1.1')
def main(args):

    logger = setup_basics(args)
    config = get_config(args)
    model = get_model(args, config)
    tokenizer = get_tokenizer(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_lr_scheduler(optimizer, args, logger)
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, config, args)