import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.DILFGFPN.FPN_basic_module import Bottleneck1D
from src.DTLGADF_module.DTLGADF_model import DTLGADF


class Proposed(nn.Module):
    def __init__(self, block, num_blocks, num_classes=5):
        super(Proposed, self).__init__()
        self.in_planes = 4

        # 第一层：DTLGADF
        self.dgadf_conv = DTLGADF(
            in_channels=1,
            out_channels=4,
            kernel_size=7,
            offset_scale=0.4,
            channel_mixing='mean',
            auto_padding=True
        )

        self.bn1 = nn.BatchNorm1d(4)

        # Bottom-up层
        self.layer1 = self._make_layer(block, 8, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 16, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 32, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)

        # 顶层
        self.toplayer = nn.Conv1d(128, 64, kernel_size=1, stride=1, padding=0)

        # 横向层
        self.latlayer1 = nn.Conv1d(64, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv1d(16, 64, kernel_size=1, stride=1, padding=0)

        # 平滑层
        self.smooth3 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        #
        self.dropout = nn.Dropout(0.3)

        # 分类头
        self.classifier = nn.Linear(32, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_normal(self, x, y):
        """
        正常上采样
        """
        _, _, L = y.size()
        x_upsampled = F.interpolate(x, size=L, mode="linear", align_corners=False)
        return x_upsampled, y

    def _upsample_add_normal(self, x, y):
        """
        线性采样
        """
        _, _, L = y.size()
        return F.interpolate(x, size=L, mode="linear", align_corners=False) + y, y

    def forward(self, x):
        # Bottom-up路径
        c1 = F.relu(self.bn1(self.dgadf_conv(x)))
        c1 = F.max_pool1d(c1, kernel_size=4)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.toplayer(c5)  # 顶层特征

        # 融合P5和C4
        up_p5, c4_aligned = self._upsample_add_normal(p5, self.latlayer1(c4))
        up_p5 = F.relu(up_p5)
        # 融合P4和C3
        up_p4, c3_aligned = self._upsample_add_normal(up_p5, self.latlayer2(c3))
        up_p4 = F.relu(up_p4)
        # 融合P3和C2
        up_p3, _ = self._upsample_add_normal(up_p4, self.latlayer3(c2))
        p2 = self.smooth3(up_p3)
        p2 = self.dropout(p2)

        # 分类头
        out = F.adaptive_avg_pool1d(p2, 1).squeeze(-1)
        out = self.classifier(out)

        # 返回分类输出和DTLGADF输出计算损失
        return out, up_p5, up_p3, c1


def init_Proposed(num_classes=5, learning_rate=0.001, weight_decay=0.01):
    """
    Initializes the model for domain adaptation with PPGAL loss.
    """
    model = Proposed(Bottleneck1D, [2, 1, 1, 1], num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    classification_criterion = nn.CrossEntropyLoss()
    return model, optimizer, classification_criterion


"""# 随机数据测试实例
Train_X = torch.randn(1050, 1, 2048)
Train_Y = torch.randint(0, 5, (1050,))
Test_X = torch.randn(350, 1, 2048)
Test_Y = torch.randint(0, 5, (350,))

train_loader = DataLoader(TensorDataset(Train_X, Train_Y), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(Test_X, Test_Y), batch_size=32, shuffle=False)

# Initialize model, optimizer, criterion
model, optimizer, criterion,_ = init_Proposed(num_classes=5)

# Train the model
train(model, train_loader, optimizer, criterion, epochs=10)

# Test the model
test(model, test_loader, criterion)"""