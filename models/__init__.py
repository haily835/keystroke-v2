from .resnet import resnet101, resnet50, resnet10
from .ctrgcn.model import CTRGCN
from .infogcn.model import InfoGCN
from .skateformer import SkateFormer
from .HyperGT import HyperGT
from .hdgcn.HDGCN import HDGCN
from .stgcn.st_gcn import STGCN
from .MyModel import MyModel
from .HCTA import HCTA
__all__ = ['resnet10', 'resnet101', 'resnet50', 'CTRGCN', 'InfoGCN', 'SkateFormer', 'HyperGT', 'HDGCN', 'STGCN', 'MyModel', 'HCTA']
