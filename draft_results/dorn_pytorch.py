import torch
import torchsummary
from torch import nn
from torchvision.models.resnet import ResNet, Bottleneck


class DORN_ResNet50_NYUV2(ResNet):

    def __init__(self, input_shape=(3, 257, 353), n_classes=68, **kwargs):
        # Equivalent to ResNet50
        self.input_shape = input_shape
        self.n_classes = n_classes
        super(DORN_ResNet50_NYUV2, self).__init__(Bottleneck, [3, 4, 6, 3], **kwargs)
        del self.avgpool
        del self.fc

        self.aspp1 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.ReLU()
        )

        for m in self.aspp1:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)

        self.aspp2 = self._make_aspp(3)
        self.aspp3 = self._make_aspp(6)
        self.aspp4 = self._make_aspp(9)

        # Encoder layers
        self.enc_avgpool = nn.AvgPool2d(3)
        self.enc_dp = nn.Dropout(0.5)
        self.enc_dense = nn.Linear(3 * 4 * 2048, 512)
        self.enc_conv = nn.Conv2d(512, 512, kernel_size=1)
        self.enc_upsample = nn.UpsamplingBilinear2d(size=(9, 12))

        self.joined = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(2560, 2048, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(2048, n_classes*2, kernel_size=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(size=input_shape[1:])
        )

    def _make_aspp(self, dilation_rate):
        # o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
        aspp = nn.Sequential(
            nn.Conv2d(2048, 512, dilation=dilation_rate, kernel_size=3, padding=dilation_rate, stride=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.ReLU()
        )

        for m in aspp:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)

        return aspp

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Encoder
        enc = self.enc_avgpool(x)
        print(enc.shape)
        enc = self.enc_dp(enc)
        enc = self.enc_dense(enc.view(enc.size(0), -1))
        enc = self.enc_conv(enc.view(enc.size(0), 512, 1, 1))
        enc = self.enc_upsample(enc)

        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspp4 = self.aspp4(x)

        x = torch.cat((enc, aspp1, aspp2, aspp3, aspp4), dim=1)
        x = self.joined(x)
        x = self.decode(x)

        return x

    def summary(self):
        torchsummary.summary(self, self.input_shape)

    def decode(self, x):
        decode_label = torch.zeros(size=[x.size(0), x.size(2), x.size(3)], dtype=torch.int64)
        for i in range(self.n_classes):
            ord_i = torch.argmax(x[:, 2 * i:2 * i + 2, :, :], dim=1)
            decode_label += ord_i
        return decode_label

model = DORN_ResNet50_NYUV2()
model.summary()
