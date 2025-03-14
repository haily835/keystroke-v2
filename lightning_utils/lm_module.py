from typing import Dict, Any
import torch
import lightning as L
import  torchmetrics
from sklearn.metrics import classification_report
import pandas as pd
from models.resnet import *
from utils.import_by_modulepath import initialize_class
import json
import os

class LmKeyClf(L.LightningModule):
    def __init__(self, 
                 model_classpath: str, 
                 model_init_args: Dict[str, Any] | None , 
                 loss_fn_classpath: str,
                 loss_fn_init_args: Dict[str, Any] | None ,
                 classes_path: str,
                 lr: float # learning rate
                ):
        super().__init__()

        # Parse classpath and model arguments
        self.model = initialize_class(model_classpath, model_init_args)
        self.loss_fn = initialize_class(loss_fn_classpath, loss_fn_init_args)
        self.lr = lr
        with open(classes_path, 'r') as f:
            self.id2label = json.load(f)
            self.label2id = {label: idx for idx, label in enumerate(self.id2label)}
        self.num_classes = len(self.id2label)
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes)
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes)
        self.test_preds = []
        self.test_targets = []

        self.train_losses = []
        self.val_losses = []
        self.save_hyperparameters()

    def forward(self, batch):
        videos, targets = batch
        preds = self.model(videos)

        if type(preds) != tuple:
            pred_ids = torch.argmax(preds, dim=1)
            loss = self.loss_fn(preds, targets)
        else:
            y_hat, z = preds
            pred_ids = torch.argmax(y_hat, dim=1)
            loss = self.loss_fn(y_hat, targets, z, self.model.z_prior)

        return loss, pred_ids

    def test_step(self, batch):
        videos, targets = batch
        loss, pred_ids = self.forward((videos, targets.long()))
        pred_labels = [self.id2label[_id] for _id in pred_ids]
        self.test_preds += pred_labels
        self.test_targets += [self.id2label[_id] for _id in targets]

        self.log('test_loss',
                 loss, 
                 sync_dist=True,
                 prog_bar=True, 
                 on_step=False)

    def on_test_end(self):
        if not os.path.exists('results'):
            os.mkdir('results')

        df = pd.DataFrame({"pred": self.test_preds, "target": self.test_targets})
        if len(self.id2label) == 2:
            df.to_csv(f'det_test_results.csv')
        else:
            df.to_csv(f'clf_test_results.csv')
        print(classification_report(self.test_targets, self.test_preds))

    def training_step(self, batch):
        videos, targets = batch        
        loss, pred_ids = self.forward((videos, targets))
        self.cur_train_acc = self.train_acc(pred_ids, targets.long())
        self.log('train_loss', loss,
                 sync_dist=True, prog_bar=True,  on_step=False, on_epoch=True)

        self.log('train_acc', self.cur_train_acc,
                 sync_dist=True, prog_bar=True, on_step=False,  on_epoch=True)
        
        return loss

    def validation_step(self, batch):
        videos, targets = batch
        loss, pred_ids = self.forward((videos, targets.long()))
        self.cur_val_acc = self.val_acc(pred_ids, targets.long())
        self.log('val_loss', loss,
                 sync_dist=True, 
                 prog_bar=True, 
                 on_step=False, 
                 on_epoch=True)
        self.log('val_acc', 
                 self.cur_val_acc,
                 sync_dist=True, 
                 prog_bar=True, 
                 on_step=False, 
                 on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        print(f"EPOCH {self.current_epoch} train_acc {self.cur_train_acc}")

    def on_validation_epoch_end(self) -> None:
        print(f"EPOCH {self.current_epoch} val_acc {self.cur_val_acc}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)