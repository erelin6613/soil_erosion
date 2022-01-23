import torch
import torch.nn as nn


def make_fcn():
    return torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.head(x)


class SegmentResNet(nn.Module):
    def __init__(self, version="resnet50", pretrained=True, activation=None):
        super().__init__()

        model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=False)
        # self.backbone = nn.Sequential(*(list(model.children())[0:4]))
        self.backbone = model
        self.activation = activation

    def forward(self, x):
        if self.activation:
            x = self.activation(self.backbone(x))
        else:
            x = self.backbone(x)
        print(x)
        return x["out"]


class SimpleSegNet(nn.Module):
    def __init__(self, in_channels, num_classes=1):

        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, 3, padding=1),
            nn.Conv2d(in_channels*2, in_channels*4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels*4, in_channels*2, 3, padding=1),
            nn.Conv2d(in_channels*2, num_classes, 3, padding=1),
            # nn.ReLU()
        )

    def forward(self, x, activation=nn.Sigmoid()):
        return activation(self.net(x))


if __name__ == '__main__':
    m = SimpleSegNet(4, 1)
    mock_img = torch.rand(1, 4, 512, 512)
    print(m(mock_img).shape)
    # print(type(m))
