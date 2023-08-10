import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', activation = 'relu'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

        self.activation = None
        if activation == 'relu':
            self.activation = F.relu
            # print("Using ReLU in BasicBlock")
        elif activation == 'gelu':
            self.activation = F.gelu
            # print("Using GeLU in BasicBlock")
        elif activation == 'elu':
            self.activation = F.elu
            # print("Using ELU in BasicBlock")
        else:
            raise Exception("Invalid activation function")

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation = 'relu'):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, activation = activation)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, activation = activation)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, activation = activation)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

        self.activation = None
        if activation == 'relu':
            self.activation = F.relu
            # print("Using ReLU in ResNet")
        elif activation == 'gelu':
            self.activation = F.gelu
            # print("Using GeLU in ResNet")
        elif activation == 'elu':
            self.activation = F.elu
            # print("Using ELU in ResNet")
        else:
            raise Exception("Invalid activation function")

    def _make_layer(self, block, planes, num_blocks, stride, activation):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes = self.in_planes, planes = planes, stride = stride, activation = activation))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20(activation = 'relu'):
    return ResNet(block = BasicBlock, num_blocks = [3, 3, 3], activation = activation)

def resnet32(activation = 'relu'):
    return ResNet(block = BasicBlock, num_blocks = [5, 5, 5], activation = activation)
