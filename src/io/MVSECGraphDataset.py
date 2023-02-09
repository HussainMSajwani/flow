from pathlib import Path
from typing import Callable, Optional
import os
from torch_geometric.data import Data, Dataset
from torch_geometric.nn.pool import radius_graph, knn_graph
from torch_geometric.transforms import Cartesian

import torch

flow_rate = 20                         #broadcasting gt flow at a rate of flow_rate Hz
dt_flow = 1/flow_rate                  #predict flow dt_flow seconds ahead
graphs_per_pred = 5                    #number of graphs to make the corr volumes
dt_graph = dt_flow / graphs_per_pred   #graphs of temporal width
downsampling_factor = 1                #only use 1/downsampling_factor of events

def make_graph(ev_arr, gt, beta=0.5e4):
    ts_sample = ev_arr[:, 3] - ev_arr[0, 3]
    ts_sample = torch.tensor(ts_sample*beta).float().reshape(-1, 1)

    coords = torch.tensor(ev_arr[:, :2]).float()
    pos = torch.hstack((ts_sample, coords))

    edge_index = knn_graph(pos, k=32)

    pol = torch.tensor(ev_arr[:, 2]).float().reshape(-1, 1)
    #feature = pol
    feature = torch.hstack((pos, pol))
    feature

    graph = Data(x=feature, edge_index=edge_index, pos=pos, y = torch.tensor(gt))
    graph = Cartesian()(graph)

    return graph

class MVSECGraphDataset(Dataset):

    def __init__(
            self, 
            root: Optional[str] = '/media/hussain/drive1/flow-data/mvsec20/MVSECGraphDatset', 
            transform: Optional[Callable] = None, 
            pre_transform: Optional[Callable] = None, 
            pre_filter: Optional[Callable] = None,
            log: bool = True
        ):
        root = Path(root).resolve()
        super().__init__(root, transform, pre_transform, pre_filter, log)
    
    def process(self):
        for h5_file in os.listdir(self.root / 'raw'):
            print(h5_file)
