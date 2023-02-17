#----------------------------------------------------------------------#
# Author: M. McEneaney, Duke University
# Date: Feb. 2023
#----------------------------------------------------------------------#

import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T

from torch_geometric.loader import DataLoader

from ..data import Dataset
from ..models import GNN
from ..train import PLModel

# Load dataset
root = "/path/to/pyg_datasets" #NOTE: DATA SHOULD BE SAVED IN <root>/processed/data.pt, create this with create_dataset.ipynb
transform = T.Compose([T.ToUndirected(),T.NormalizeFeatures()])
dataset = Dataset(root, transform=transform, pre_transform=None, pre_filter=None)

# Sanity check
data = dataset[0]
print(data.x)
print(dataset)
if transform is not None:
    print(transform(dataset[0]).edge_index)
    print(transform(dataset[0]).x)

# Put model on device if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Split dataset
torch.manual_seed(12345)
dataset = dataset.shuffle()

print(len(dataset))

fracs = [0.8, 0.1, 0.1] #NOTE: SHOULD CHECK np.sum(fracs) == 1 and len(fracs)==3
fracs = [torch.sum(torch.tensor(fracs[:idx])) for idx in range(1,len(fracs)+1)]
print(fracs)
split1, split2 = [int(len(dataset)*frac) for frac in fracs[:-1]]
train_dataset = dataset[:split1]
val_dataset = dataset[split1:split2]
test_dataset = dataset[split2:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of validation graphs: {len(val_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

# Create dataloaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()
    break

# Define training and validation routines
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(loader):
    model.train()
    for data in loader:  # Iterate in batches over the training dataset.
        counts = torch.unique(data.y,return_counts=True)[1]
        weights = counts / len(data.y)
        weights = np.power(weights,-1)
        weight = torch.tensor([weights[idx] for idx in torch.squeeze(data.y)]).to(device)
        criterion = torch.nn.BCELoss(weight=weight)
        
        data = data.to(device)#NOTE: ADDED
        out = torch.squeeze(model(data.x, data.edge_index, data.batch))  # Perform a single forward pass.
        loss = criterion(out, data.y.float())  # Compute the loss.
        
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

@torch.no_grad()
def val(loader):
    model.eval()

    correct = 0
    loss_tot = 0.0
    for data in loader:  # Iterate in batches over the training/test dataset.
        counts = torch.unique(data.y,return_counts=True)[1]
        weights = counts / len(data.y)
        weights = np.power(weights,-1)
        weight = torch.tensor([weights[idx] for idx in torch.squeeze(data.y)]).to(device)
        criterion = torch.nn.BCELoss(weight=weight)
        
        data = data.to(device)
        out = torch.squeeze(model(data.x, data.edge_index, data.batch))
        loss = criterion(out, data.y.float())
#         pred = out.argmax(dim=1)  # Use the class with highest probability.
        pred = out.round() #NOTE: JUST FOR USING BCELOSS -> ARGMAX COLLAPSES TO A ONE ELEMENT TENSOR
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        loss_tot += loss.item()
    return correct / len(loader.dataset), loss_tot / len(loader.dataset)  # Derive ratio of correct predictions.

# Train and test the model
nepochs = 5
for epoch in range(1, nepochs+1):
    train(train_loader)
    train_acc, train_loss = val(train_loader)
    val_acc, val_loss = val(val_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f} Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f} Loss: {val_loss:.4f}')
    
test_acc, test_loss = val(test_loader)
print(f'Test Acc: {train_acc:.4f} Loss: {train_loss:.4f}')
