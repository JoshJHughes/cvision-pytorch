from torch import nn

class Lenet_Custom(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(out_features=120),
            nn.Sigmoid(),
            nn.LazyLinear(out_features=84),
            nn.Sigmoid(),
            nn.LazyLinear(out_features=num_classes)
        )

    def forward(self, x):
        logits = self.net(x)
        return logits