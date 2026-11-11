from torch import nn

class ImageNetAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (Downsampling layers)
        self.down1 = nn.Conv2d(1, 64, 3, stride=2, padding=1)  # Input 1 channel for grayscale
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.down4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

        # Decoder (Upsampling layers)
        self.up1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.up4 = nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1)  # Output 3 channels for RGB

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder path
        d1 = self.relu(self.down1(x))
        d2 = self.relu(self.down2(d1))
        d3 = self.relu(self.down3(d2))
        d4 = self.relu(self.down4(d3))

        # Decoder path
        u1 = self.relu(self.up1(d4))
        u2 = self.relu(self.up2(u1))
        u3 = self.relu(self.up3(u2))
        u4 = self.sigmoid(self.up4(u3))

        return u4
