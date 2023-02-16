#----------------------------------------------------------------------#
# Author: M. McEneaney
# Institution: Duke University
# Date: Feb. 2023
#----------------------------------------------------------------------#

import torch
from torch_geometric.data import InMemoryDataset, download_url

class Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, data_list=None, verbose=True):
        self.data_list = data_list
        super().__init__(root, transform, pre_transform, pre_filter)
        if verbose: print("INFO: dataset savid in",self.processed_paths)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = self.data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
