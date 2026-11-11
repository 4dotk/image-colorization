from torch import nn, cat

class AutoEncoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.down1 = nn.Conv2d(1, 64, 3, stride = 2)
    self.down2 = nn.Conv2d(64, 128, 3, stride = 2, padding = 1)
    self.down3 = nn.Conv2d(128, 256, 3, stride = 2, padding = 1)
    self.down4 = nn.Conv2d(256, 512, 3, stride = 2, padding = 1)

    self.up1 = nn.ConvTranspose2d(512, 256, 3, stride = 2, padding = 1)
    self.up2 = nn.ConvTranspose2d(512, 128, 3, stride = 2, padding = 1)
    self.up3 = nn.ConvTranspose2d(256, 64, 3, stride = 2, padding = 1, output_padding = 1)
    self.up4 = nn.ConvTranspose2d(128, 3, 3, stride = 2, output_padding = 1)

    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    d1 = self.relu(self.down1(x))
    d2 = self.relu(self.down2(d1))
    d3 = self.relu(self.down3(d2))
    d4 = self.relu(self.down4(d3))

    u1 = self.relu(self.up1(d4))
    u2 = self.relu(self.up2(cat((u1, d3), dim = 1)))
    u3 = self.relu(self.up3(cat((u2, d2), dim = 1)))
    u4 = self.sigmoid(self.up4(cat((u3, d1), dim = 1)))

    return u4

