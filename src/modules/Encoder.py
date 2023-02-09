import torch.nn as nn
from torch_geometric.nn import SplineConv
from torch_geometric.transforms import Cartesian
from .MaxPooling import MaxPooling, MaxPoolingX

class Encoder(nn.Module):

    def __init__(
            self, 
            n_feature=4
            ) -> None:
        super().__init__()

        pseudo = Cartesian(norm=True, cat=False)

        self.conv1 = SplineConv(4, 32, dim=3, kernel_size=2)
        self.norm1 = nn.BatchNorm1d(32)

        self.conv2 = SplineConv(32, 64, dim=3, kernel_size=2)
        self.norm2 = nn.BatchNorm1d(64)
        self.pool2 = MaxPooling((4, 3), transform=pseudo)

        self.conv3 = SplineConv(64, 64, dim=3, kernel_size=2)
        self.norm3 = nn.BatchNorm1d(64)
        
        self.conv4 = SplineConv(64, 64, dim=3, kernel_size=2)
        self.norm4 = nn.BatchNorm1d(64)
        self.pool4 = MaxPooling((16, 12), transform=pseudo)

        self.conv5 = SplineConv(64, 128, dim=3, kernel_size=2)
        self.norm5 = nn.BatchNorm1d(128)

        self.conv6 = SplineConv(128, 256, dim=3, kernel_size=2)
        self.norm6 = nn.BatchNorm1d(256)
        

    def forward(self, data):
        data.x = nn.functional.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm1(data.x)

        data.x = nn.functional.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2(data.x)
        data = self.pool2(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        data.x = nn.functional.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm3(data.x)

        data.x = nn.functional.elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm4(data.x)
        data = self.pool4(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        data.x = nn.functional.elu(self.conv5(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm5(data.x)

        data.x = nn.functional.elu(self.conv6(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm6(data.x)

        return data