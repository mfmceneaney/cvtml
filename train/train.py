#----------------------------------------------------------------------#
# Author: M. McEneaney, Duke University
# Date: Feb. 2023
#----------------------------------------------------------------------#

import torch
from torchmetrics import Accuracy
from torch.utils.data import random_split
import pytorch_lightning as pl

class PLModel(pl.LightningModule):
    """
    This is just a pytorch lightning model wrapper for a pytorch binary
    classification model.
    """

    def __init__(self,
                 model_class = None,
                 model_class_args = [],
                 model_class_kwargs = {},
                 criterion = None,
                 optimizer = None,
                 optimizer_kwargs = None,
                 task = "binary",
                 num_classes = 1,
                 weight = True,
                 dataset_class = None,
                 ds_args = [],
                 ds_kwargs = {},
                 lengths = [1.0],
                 dataloader_class = None,
                 train_batch_size = 64,
                 val_batch_size = 64,
                 test_batch_size = 64,
                 num_workers = 4
                ):
        super(PLModel, self).__init__()
        self.criterion = criterion if criterion is not None else F.binary_cross_entropy
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.task = task
        if self.task!='binary': raise TypeError('PLModel: Only binary classification implemented so far')
        self.num_classes = num_classes #NOTE: FOR BCELoss SHOULD HAVE NUM_CLASSES=1.
        self.weight = weight #NOTE: Whether or not to use loss weighting on batch basis
        self.dataset_class = dataset_class
        self.ds_args = ds_args
        self.ds_kwargs = ds_kwargs
        self.lengths = lengths
        self.dataloader_class = dataloader_class
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        
        # Init random class attributes
        self.dataset = None
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None
        
        self.train_accuracy = Accuracy(task=self.task, num_classes=self.num_classes)
        self.val_accuracy = Accuracy(task=self.task, num_classes=self.num_classes)
        self.test_accuracy = Accuracy(task=self.task, num_classes=self.num_classes)
        
        self.model = model_class(*model_class_args,**model_class_kwargs)

    def training_step(self, batch, batch_idx):
        x = torch.squeeze(self.model(batch.x, batch.edge_index, batch.batch))
        counts = torch.pow(torch.unique(batch.y,return_counts=True)[1] / len(batch.y), -1) if self.weight else None
        weight = torch.tensor([counts[idx] for idx in torch.squeeze(batch.y)]).to(x.device) if self.weight else None #NOTE: THIS ONLY WORKS FOR BINARY CLASSIFICATION WITH BCELOSS
        loss = self.criterion(x, batch.y.float(), weight=weight)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, prog_bar=True, batch_size=self.train_batch_size)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = torch.squeeze(self.model(batch.x, batch.edge_index, batch.batch))
        counts = torch.pow(torch.unique(batch.y,return_counts=True)[1] / len(batch.y), -1) if self.weight else None
        weight = torch.tensor([counts[idx] for idx in torch.squeeze(batch.y)]).to(x.device) if self.weight else None #NOTE: THIS ONLY WORKS FOR BINARY CLASSIFICATION WITH BCELOSS
        loss = self.criterion(x, batch.y.float(), weight=weight)
        preds = x.round() #NOTE: ONLY USE FOR BINARY CLASSIFICATION
        self.val_accuracy.update(preds, batch.y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True, batch_size=self.val_batch_size)
        self.log('val_acc', self.val_accuracy, prog_bar=True, batch_size=self.val_batch_size)
        return loss
    
    @torch.no_grad()
    def test_step(self, batch, batch_nb):
        x = torch.squeeze(self.model(batch.x, batch.edge_index, batch.batch))
        counts = torch.pow(torch.unique(batch.y,return_counts=True)[1] / len(batch.y), -1) if self.weight else None
        weight = torch.tensor([counts[idx] for idx in torch.squeeze(batch.y)]).to(x.device) if self.weight else None #NOTE: THIS ONLY WORKS FOR BINARY CLASSIFICATION WITH BCELOSS
        loss = self.criterion(x, batch.y.float(), weight=weight)
        preds = x.round() #NOTE: ONLY USE FOR BINARY CLASSIFICATION
        self.val_accuracy.update(preds, batch.y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('test_loss', loss, prog_bar=True, batch_size=self.test_batch_size)
        self.log('test_acc', self.test_accuracy, prog_bar=True, batch_size=self.test_batch_size)
        return loss

    def configure_optimizers(self):
        if self.optimizer is not None and self.optimizer_kwargs is not None:
            return self.optimizer(self.parameters(), **self.optimizer_kwargs)
        else:
            return torch.optim.Adam(self.parameters(), lr=0.01)
            
    def prepare_data(self): #NOTE: DO NOT MAKE ANY STATE ASSIGNMENTS HERE, JUST DOWNLOAD THE DATA IF NEEDED
        pass

    def setup(self, stage=None): #NOTE: THIS RUNS ACROSS ALL GPUS
        # Assign train/val/test datasets for use in dataloaders
        if self.dataset is None:
            self.dataset = self.dataset_class(*self.ds_args,**self.ds_kwargs) #NOTE: NEEDED KWARGS datasetclass ds_args, ds_kwargs, lengths
        if len(self.lengths)==2 and self.ds_train is None and self.ds_val is None:
            self.ds_train, self.ds_val = random_split(self.dataset, self.lengths)
        elif len(self.lengths)==3 and self.ds_train is None and self.ds_val is None:
            self.ds_train, self.ds_val, self.ds_test = random_split(self.dataset, self.lengths)

    def train_dataloader(self):
        return self.dataloader_class(self.ds_train, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers) #NOTE: NEEDED KWARGS dataloader_class train_batch_size val test...

    def val_dataloader(self):
        return self.dataloader_class(self.ds_val, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return self.dataloader_class(self.ds_test, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers)
