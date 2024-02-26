import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, num_classes: int, n_channels: int, pretrained: bool):
        """Initializes ResNet18. Modifies input and output layer.

        Args:
            n_classes (int): Number of classes.
            n_channels (int): Number of channels.
            pretrained (bool): If true, loads pre-trained weights.
        """
        super().__init__()
        self.resnet18 = models.resnet18(num_classes=num_classes, pretrained=pretrained)
        # Define input layer with corresponding number of input channels (gray-scaled or RGB)
        self.resnet18.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.resnet18(x)
        return x


class DeeperCNN(torch.nn.Module):
    def __init__(self, output_dim: int, in_channels: int):
        super(DeeperCNN, self).__init__()
        self.conv_layer = torch.nn.Sequential(
            # Conv Layer block 1
            torch.nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            # torch.nn.Dropout(p=0.25)
        )

        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(7744, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, output_dim),
        )

    def forward(self, x):
        """Perform forward."""
        # x = torch.unsqueeze(x, 1)
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


class SimpleCNN(torch.nn.Module):
    def __init__(self, output_dim: int, in_channels: int):
        super(SimpleCNN, self).__init__()
        self.output_dim = output_dim
        self.in_channels = in_channels
        self.conv_layer = torch.nn.Sequential(
            # Conv Layer block 1
            torch.nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=3, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.25),
        )

        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(9216, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, output_dim),
        )

    def forward(self, x):
        """Perform forward."""
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleNet(nn.Module):
    def __init__(self, n_channels: int, bilinear: bool = True, apply_sigmoid: bool = False):
        super(SimpleNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.apply_sigmoid = apply_sigmoid

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.out_conv = OutConv(64, 1)
        self.lin = nn.Linear(32 * 32, 784)

    def forward(self, x):
        x1 = self.inc(x)  # Output shape: Channel=64, Width=28, Height=28
        x2 = self.down1(x1)  # Channel=128, Width=14, Height=14
        x3 = self.down2(x2)  # Channel=256, Width=7, Height=7
        x4 = self.down3(x3)  # Channel=512, Width=3, Height=3
        x5 = self.down4(x4)  # Channel=512, Width=1, Height=1
        x = self.up1(x5, x4)  # Channel=256, Width=3, Height=3
        x = self.up2(x, x3)  # Channel=128, Width=7, Height=7
        x = self.up3(x, x2)  # Channel=64, Width=14, Height=14
        x = self.up4(x, x1)  # Channel=128, Width=28, Height=28
        logits = self.out_conv(x)  # N, C, W, H
        # a = torch.abs(logits[:, 0, :, :])
        # b = torch.abs(logits[:, 1, :, :])
        # logits = torch.unsqueeze(a / (a + b), dim=1)
        # return shape: N, C, W, H
        # return torch.sigmoid(logits)
        return torch.sigmoid(logits) if self.apply_sigmoid else logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UCICensusClassifier(nn.Module):
    def __init__(self, input_size, output_size=3, num_layers=1, hidden_size=50):
        """Simple nn to classify categorical data.

        Args:
            input_size (int): Number of total input dimensions
            output_size (int): Number of Target Classes
            factor (int): Factor to increase the number of layers
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.linears = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.linears.extend([nn.Linear(hidden_size, hidden_size) for i in range(1, self.num_layers - 1)])
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x_input):
        x = x_input.flatten(start_dim=1)
        for l in self.linears:
            x = F.relu(l(x))
        x = self.out(x)
        return x


class SaliencyModel(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(SaliencyModel, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block=block, planes=64, num_blocks=num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block=block, planes=128, num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block=block, planes=256, num_blocks=num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block=block, planes=512, num_blocks=num_blocks[3], stride=2)

        self.uplayer4 = UpSampleBlock(in_channels=512, out_channels=256, passthrough_channels=256)
        self.uplayer3 = UpSampleBlock(in_channels=256, out_channels=128, passthrough_channels=128)
        self.uplayer2 = UpSampleBlock(in_channels=128, out_channels=64, passthrough_channels=64)

        self.embedding = nn.Embedding(num_classes, 512)
        # self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.saliency_chans = nn.Conv2d(64, 2, kernel_size=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, labels):
        out = F.relu(self.bn1(self.conv1(x)))

        scale1 = self.layer1(out)
        scale2 = self.layer2(scale1)
        scale3 = self.layer3(scale2)
        scale4 = self.layer4(scale3)

        em = torch.squeeze(self.embedding(labels.view(-1, 1)), 1)
        act = torch.sum(scale4 * em.view(-1, 512, 1, 1), 1, keepdim=True)
        th = torch.sigmoid(act)
        scale4 = scale4 * th

        upsample3 = self.uplayer4(scale4, scale3)
        upsample2 = self.uplayer3(upsample3, scale2)
        upsample1 = self.uplayer2(upsample2, scale1)

        saliency_chans = self.saliency_chans(upsample1)

        # out = F.avg_pool2d(scale4, 4)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)

        a = torch.abs(saliency_chans[:, 0, :, :])
        b = torch.abs(saliency_chans[:, 1, :, :])

        return torch.unsqueeze(a / (a + b), dim=1)


def saliency_model():
    return SaliencyModel(Block, [2, 2, 2, 2])


class UpSampleBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, passthrough_channels, stride=1):
        super(UpSampleBlock, self).__init__()
        self.upsampler = SubpixelUpsampler(in_channels=in_channels, out_channels=out_channels)
        self.follow_up = Block(out_channels + passthrough_channels, out_channels)

    def forward(self, x, passthrough):
        out = self.upsampler(x)
        out = torch.cat((out, passthrough), 1)
        return self.follow_up(out)


def SubpixelUpsampler(
    in_channels, out_channels, kernel_size=3, activation_fn=lambda: torch.nn.ReLU(inplace=False), follow_with_bn=True
):
    _modules = [
        CNNBlock(in_channels, out_channels * 4, kernel_size=kernel_size, follow_with_bn=follow_with_bn),
        PixelShuffleBlock(),
        activation_fn(),
    ]
    return nn.Sequential(*_modules)


class PixelShuffleBlock(nn.Module):
    def forward(self, x):
        return F.pixel_shuffle(x, 2)


def CNNBlock(
    in_channels,
    out_channels,
    kernel_size=3,
    layers=1,
    stride=1,
    follow_with_bn=True,
    activation_fn=lambda: nn.ReLU(True),
    affine=True,
):
    assert layers > 0 and kernel_size % 2 and stride > 0
    current_channels = in_channels
    _modules = []
    for layer in range(layers):
        _modules.append(
            nn.Conv2d(
                current_channels,
                out_channels,
                kernel_size,
                stride=stride if layer == 0 else 1,
                padding=int(kernel_size / 2),
                bias=not follow_with_bn,
            )
        )
        current_channels = out_channels
        if follow_with_bn:
            _modules.append(nn.BatchNorm2d(current_channels, affine=affine))
        if activation_fn is not None:
            _modules.append(activation_fn())
    return nn.Sequential(*_modules)


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MyWideResNet(nn.Module):
    def __init__(
        self,
        reduce_kernel_size=True,
        binary_classification=False,
        remove_batch_norm=True,
        add_mask_channel=False,
        imagenet_pretrained=True,
    ):
        """Initialize the model

        Args:
            reduce_kernel_size (bool): Reduce the kernel size of the first layer to 3x3
            binary_classification (bool): Use a single output neuron for binary classification
            remove_batch_norm (bool): Remove the batch normalization layers
            add_mask_channel (bool): Add a mask channel to the input
            imagenet_pretrained (bool): Load the imagenet pretrained weights
        """

        super(MyWideResNet, self).__init__()
        weights = models.Wide_ResNet50_2_Weights.DEFAULT if imagenet_pretrained is True else None
        # Load the wide ResNet-28-10 model
        self.model = models.wide_resnet50_2(weights)

        if reduce_kernel_size is True:
            # Reduce Kernel Size since we have smaller images
            if add_mask_channel is True:
                self.model.conv1 = torch.nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
            else:
                self.model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if binary_classification is True:
            self.model.fc = torch.nn.Linear(2048, 1)
        else:
            self.model.fc = torch.nn.Linear(2048, 3)

        if remove_batch_norm is True:
            # Remove batch normalization layers
            bn_modules = [name for name, module in self.model.named_modules() if isinstance(module, nn.BatchNorm2d)]
            for name in bn_modules:
                setattr(self.model, name, nn.Identity())

    def forward(self, x):
        x = self.model(x)
        return x
