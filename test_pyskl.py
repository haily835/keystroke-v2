from models.pyskl.models.cnns import ResNet3d
from models.pyskl.models.heads import SimpleHead
import torch


if __name__ == "__main__":
    model = ResNet3d()
    clf_head = SimpleHead(num_classes=30, in_channels=2048)
    x = torch.rand(size=(4, 3, 8, 21, 2))
    x = model(x)
    out = clf_head(x)
    print(out.shape)