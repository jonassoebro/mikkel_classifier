import torchvision.models as models
from torch.nn import Module, Sequential, Linear

class ResNet18(Module):

    def __init__(self, pretrained: bool = False, in_dim: int = 512, out_dim: int = 1):
        super(ResNet18, self).__init__()
        self.resnet = Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])
        self.linear = Linear(in_features=in_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.linear(x)
        return x
