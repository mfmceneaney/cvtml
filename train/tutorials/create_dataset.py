#----------------------------------------------------------------------#
# Author: M. McEneaney, Duke University
# Date: Feb. 2023
#----------------------------------------------------------------------#

import hipopy.hipopy as hp
import numpy.ma as ma
from tqdm import tqdm
import torch
from torch_geometric.data import Data

from ..data import Dataset

# Set params for reading data
files = "/work/clas12/jnp/cvt/out_gen_cvt_rgbbg50na.hipo"
banks = [
    "BST::HitsPos"
]
keys = [
    'BST::HitsPos_ID',
    'BST::HitsPos_sector',
    'BST::HitsPos_layer',
    'BST::HitsPos_strip',
    'BST::HitsPos_r1',
    'BST::HitsPos_theta1',
    'BST::HitsPos_phi1',
    'BST::HitsPos_r2',
    'BST::HitsPos_theta2',
    'BST::HitsPos_phi2',
    'BST::HitsPos_tstatus',
    'BST::HitsPos_rstatus'
]
step = 1
max_events = 10**2
label_keys = keys[10:]
data_keys = keys[4:10]
datalist = []

# Loop HIPO files to create graphs
for idx, batch in tqdm(enumerate(hp.iterate(files,banks=banks,step=step,experimental=True))): #NOTE: RETURNS ARRAYS WITH SHAPE (NEVENTS=1,NROWS)
    
    # Check labels tstatus and rstatus
    rec_mask = torch.eq(batch[label_keys[1]][0],1) #NOTE: ONLY GET TRACKS THAT ARE RECONSTRUCTED
    y = torch.tensor(ma.array(batch[label_keys[0]][0])[rec_mask], dtype=torch.long) #NOTE: IF YOU WANT TO DO NODE CLASSIFICATION
    y = torch.tensor([1 if y.sum().item()==y.shape[0] else 0],dtype=torch.long) #NOTE: IF YOU JUST WANT ALL THE TRACKS TO BE TRUE

    # Get data arrays
    x = torch.moveaxis(torch.tensor([ma.array(batch[key][0])[rec_mask] for key in data_keys], dtype=torch.float),[0,1],[1,0])
    nnodes = x.shape[0] #NOTE: AFTER TORCH.MOVEAXIS ABOVE x.shape should = (NNODES,NFEATURES)
    if nnodes<=0: continue
    
    # Create graph
    edge_index = torch.tensor([[i for i in range(nnodes)],[i+1 if i<nnodes-1 else 0 for i in range(nnodes)]],dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Sanity check
    if torch.max(edge_index)>=nnodes or torch.min(edge_index)<0:
        print("DEBUGGING: ERROR: torch.max(edge_index)>=nnodes")
        print("DEBUGGING: edge_index = ",edge_index)
        print("DEBUGGING: nnodes = ",nnodes)
        print("DEGBUGGING: idx = ",idx)
    if edge_index.shape[1]!=nnodes:
        print("DEBUGGING: ERROR: edge_index.shape[1]!=nnodes")
        print("DEBUGGING: edge_index = ",edge_index)
        print("DEBUGGING: nnodes = ",nnodes)
        print("DEGBUGGING: idx = ",idx)
    
    # Add graph to list
    datalist.append(data)

# Create dataset
root = "pyg_datasets" #NOTE: DATA WILL BE SAVED IN <root>/processed/data.pt
mydataset = MyOwnDataset(root, transform=None, pre_transform=None, pre_filter=None, data_list=data_list)
print(mydataset[0])
print(len(mydataset))#NOTE: YOU SHOULD SEE the Processing...\nDone! message from dataset.process() being called.
