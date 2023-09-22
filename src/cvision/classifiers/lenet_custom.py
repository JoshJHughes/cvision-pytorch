from torch import nn

class Lenet_Custom(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.activation_fn = nn.Tanh
        self.pool_fn = nn.AvgPool2d

        self.features = nn.Sequential(
            nn.LazyConv2d(out_channels=6, kernel_size=5, padding=2),
            self.activation_fn(),
            self.pool_fn(kernel_size=2, stride=2),

            nn.LazyConv2d(out_channels=16, kernel_size=5),
            self.activation_fn(),
            self.pool_fn(kernel_size=2, stride=2),
        )

        self.classifier= nn.Sequential(
            nn.Flatten(),

            nn.LazyLinear(out_features=120),
            self.activation_fn(),

            nn.LazyLinear(out_features=84),
            self.activation_fn(),

            nn.LazyLinear(out_features=num_classes),
        )

    def forward(self, x):
        net = nn.Sequential(self.features, self.classifier)
        logits = net(x)
        return logits