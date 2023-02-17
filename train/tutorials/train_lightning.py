#----------------------------------------------------------------------#
# Author: M. McEneaney, Duke University
# Date: Feb. 2023
#----------------------------------------------------------------------#

import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import MLFlowLogger

from ..models import GNN
from ..data import Dataset
from ..train import PLModel

# Define pytorch lightning model with all the training/validation parameters
transform = T.Compose([T.ToUndirected(),T.NormalizeFeatures()])
plmodel = PLModel(
         model_class = GNN,
         model_class_args = [],
         model_class_kwargs = {'in_channels':dataset.num_node_features,'hidden_channels':64,'out_channels':1},
         criterion = torch.nn.functional.binary_cross_entropy,
         optimizer = torch.optim.Adam,
         optimizer_kwargs = {'lr':0.01},
         task = 'binary',
         num_classes = 1,
         weight = True,
         dataset_class = Dataset,
         ds_args = ['/work/clas12/users/mfmce/pyg_datasets/'],
         ds_kwargs = {'transform':transform, 'pre_transform':None, 'pre_filter':None},
         lengths = [0.8,0.1,0.1],
         dataloader_class = DataLoader,
         train_batch_size = 16,
         val_batch_size = 16,
         test_batch_size = 16,
         num_workers = 4
        )

# Sanity check
print(type(plmodel))
print(type(plmodel.model))

# Train model
pl.seed_everything(72, workers=True)
use_mlflow = False
mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs") if use_mlflow else None
trainer = Trainer(
    default_root_dir="./", #NOTE: PL AUTOMATICALLY SAVES PL CHECKPOINT TO PWD UNLESS THIS OPTION IS DIFFERENT
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,
    max_epochs=3,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=CSVLogger(save_dir="logs/") if mlflow_logger is None else mlf_logger,
    deterministic=True, #NOTE: For reproducibility use pytorch_lightning.seed_everything and this
    logger=mlf_logger
)
trainer.fit(plmodel)

# Test model - pl automatically saves best and last checkpoints
trainer.test(ckpt='best')

# save for use in production environment
script = plmodel.to_torchscript() #NOTE: Different method for pl
torch.jit.save(script, "model.pt")
