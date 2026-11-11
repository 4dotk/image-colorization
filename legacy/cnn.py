from torch import nn, sigmoid

class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv8 = nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv9 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv10 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=4, dilation=2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.relu(self.conv5(x))
        x = nn.functional.relu(self.conv6(x))
        x = nn.functional.relu(self.conv7(x))
        x = nn.functional.relu(self.conv8(x))
        x = nn.functional.relu(self.conv9(x))
        x = sigmoid(self.conv10(x))
        return x
