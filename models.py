import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class MLP(nn.Module):

    def __init__(self, input_dim=1, hidden_dim=100, output_dim=1,
                 activation_hidden=torch.relu, activation_output=torch.sigmoid,
                 initialization=nn.init.kaiming_normal_,
                 mu=None, std=None, bias=True):
        super(MLP, self).__init__()

        self.activation_hidden = activation_hidden
        self.activation_output = activation_output

        self.first_layer = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.second_layer = nn.Linear(hidden_dim, output_dim, bias=bias)

        if initialization == nn.init.kaiming_normal_:
            initialization(self.first_layer.weight)
            initialization(self.second_layer.weight)
        elif initialization == nn.init.normal_:
            mu = 0.0 if mu is None else mu
            std = 1.0 if std is None else std
            initialization(self.first_layer.weight, mu, std)
            initialization(self.second_layer.weight, mu, std)

    def forward(self, x):
        self.first_layer_output = self.activation_hidden(self.first_layer(x))
        return self.activation_output(self.second_layer(self.first_layer_output))


class LambdaLayer(nn.Module):
    def __init__(self, planes):
        super(LambdaLayer, self).__init__()
        self.planes = planes

    def forward(self, x):
        return F.pad(x[:, :, ::2, ::2], [0, 0, 0, 0, self.planes // 4, self.planes // 4], "constant", 0)


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


def get_model_object_from_lst_param_name(m, lst):
    if len(lst) == 1:
        return m.__getattr__(lst[0])
    return get_model_object_from_lst_param_name(m.__getattr__(lst[0]), lst[1:])


def get_model_object_from_lst_buffer_name(m, lst):
    if len(lst) == 1:
        return m.__getattr__(lst[0])
    return get_model_object_from_lst_buffer_name(m.__getattr__(lst[0]), lst[1:])


def find_param_names(m):
    temp_lst = list(map(lambda x: x[0].split(".")[:-1], m.named_parameters()))
    lst = []
    for t in temp_lst:
        if t in lst or "bn" in t[-1]:
            continue
        lst.append(t)
    return lst


def find_buffer_names(m):
    temp_lst = list(map(lambda x: x[0].split(".")[:-1], m.named_buffers()))
    lst = []
    for t in temp_lst:
        if t in lst:
            continue
        lst.append(t)
    return lst


def generate_dict_params_buffs(m, lst_names, get_model_object, coef, grads):
    name = ".".join(lst_names)
    params_dict = dict(get_model_object(m, lst_names).named_parameters())
    for key in params_dict:
        if key in ["weight", "bias"]:
            if f"{name}.{key}" in dict(m.named_parameters()).keys():
                idx = list(map(lambda n: n[0], m.named_parameters())).index(f"{name}.{key}")

                params_dict[key] = params_dict[key].data - coef * grads[idx]
            else:
                params_dict[key] = params_dict[key].data

    buffs_dict = dict(get_model_object(m, lst_names).named_buffers())
    for key in buffs_dict:
        if key in ["running_mean", "running_var"]:
            buffs_dict[key] = buffs_dict[key].data
    if "num_batches_tracked" in buffs_dict:
        del buffs_dict["num_batches_tracked"]
    return params_dict, buffs_dict


class F_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(F_BasicBlock, self).__init__()

        self.conv1 = {"stride": stride, "padding": 1, "bias": None}
        self.conv2 = {"stride": 1, "padding": 1, "bias": None}

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(planes)

    def forward(self, x, m, lst_param_names, lst_buffer_names, coef, grads):
        params_conv1, buffs_conv1 = generate_dict_params_buffs(m, lst_param_names[0],
                                                               get_model_object_from_lst_param_name, coef, grads)
        out = F.conv2d(input=x, **params_conv1, **buffs_conv1, **self.conv1)
        params_bn1, buffs_bn1 = generate_dict_params_buffs(m, lst_buffer_names[0],
                                                           get_model_object_from_lst_buffer_name, coef, grads)
        out = F.batch_norm(input=out, training=True, **params_bn1, **buffs_bn1)
        out = F.relu(out)
        params_conv2, buffs_conv2 = generate_dict_params_buffs(m, lst_param_names[1],
                                                               get_model_object_from_lst_param_name, coef, grads)
        out = F.conv2d(input=out, **params_conv2, **buffs_conv2, **self.conv2)
        params_bn2, buffs_bn2 = generate_dict_params_buffs(m, lst_buffer_names[1],
                                                           get_model_object_from_lst_buffer_name, coef, grads)
        out = F.batch_norm(input=out, training=True, **params_bn2, **buffs_bn2)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class F_ResNet_32(nn.Module):

    def __init__(self, block, num_blocks):
        super(F_ResNet_32, self).__init__()
        self.in_planes = 16

        self.conv1 = {"stride": 1, "padding": 1, "bias": None}

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = nn.ModuleList()
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return layers

    def forward(self, x, m, coef, grads):
        lst_param_names = find_param_names(m)
        lst_buffer_names = find_buffer_names(m)

        params_conv1, buffs_conv1 = generate_dict_params_buffs(m, lst_param_names[0],
                                                               get_model_object_from_lst_param_name, coef, grads)
        out = F.conv2d(input=x, **params_conv1, **buffs_conv1, **self.conv1)
        params_bn1, buffs_bn1 = generate_dict_params_buffs(m, lst_buffer_names[0],
                                                           get_model_object_from_lst_buffer_name, coef, grads)
        out = F.batch_norm(input=out, training=True, **params_bn1, **buffs_bn1)
        out = F.relu(out)

        layer1_lst_params_names = list(filter(lambda l: l[0] == "layer1", lst_param_names))
        layer1_lst_buffers_names = list(filter(lambda l: l[0] == "layer1", lst_buffer_names))
        for layer in range(len(self.layer1)):
            layer_lst_params = list(filter(lambda l: l[1] == str(layer), layer1_lst_params_names))
            layer_lst_buffers = list(filter(lambda l: l[1] == str(layer), layer1_lst_buffers_names))
            out = self.layer1[layer](out, m, layer_lst_params, layer_lst_buffers, coef, grads)

        layer1_lst_params_names = list(filter(lambda l: l[0] == "layer2", lst_param_names))
        layer1_lst_buffers_names = list(filter(lambda l: l[0] == "layer2", lst_buffer_names))
        for layer in range(len(self.layer2)):
            layer_lst_params = list(filter(lambda l: l[1] == str(layer), layer1_lst_params_names))
            layer_lst_buffers = list(filter(lambda l: l[1] == str(layer), layer1_lst_buffers_names))
            out = self.layer2[layer](out, m, layer_lst_params, layer_lst_buffers, coef, grads)

        layer1_lst_params_names = list(filter(lambda l: l[0] == "layer3", lst_param_names))
        layer1_lst_buffers_names = list(filter(lambda l: l[0] == "layer3", lst_buffer_names))
        for layer in range(len(self.layer3)):
            layer_lst_params = list(filter(lambda l: l[1] == str(layer), layer1_lst_params_names))
            layer_lst_buffers = list(filter(lambda l: l[1] == str(layer), layer1_lst_buffers_names))
            out = self.layer3[layer](out, m, layer_lst_params, layer_lst_buffers, coef, grads)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        params_linear, buffs_linear = generate_dict_params_buffs(m, lst_param_names[-1],
                                                                 get_model_object_from_lst_param_name, coef, grads)
        out = F.linear(input=out, **params_linear, **buffs_linear)
        return out


"""## Resnet-32"""


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet32(num_classes=10):
    return ResNet(BasicBlock,
                  num_classes=num_classes,
                  num_blocks=[5, 5, 5])


"""## Wide-Resnet-28-10"""


class F_W_BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super(F_W_BasicBlock, self).__init__()

        self.conv1 = {"stride": stride, "padding": 1, "bias": None}
        self.conv2 = {"stride": 1, "padding": 1, "bias": None}
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and {"stride": stride, "padding": 0, "bias": None} or None

    def forward(self, x, m, lst_param_names, lst_buffer_names, coef, grads):
        params_bn1, buffs_bn1 = generate_dict_params_buffs(m, lst_buffer_names[0],
                                                           get_model_object_from_lst_buffer_name, coef, grads)
        temp = F.batch_norm(input=x, training=True, **params_bn1, **buffs_bn1)

        params_conv1, buffs_conv1 = generate_dict_params_buffs(m, lst_param_names[0],
                                                               get_model_object_from_lst_param_name, coef,
                                                               grads)

        if not self.equalInOut:
            x = F.relu(temp, inplace=True)
            out = F.conv2d(input=x, **params_conv1, **buffs_conv1, **self.conv1)
        else:
            out = F.relu(temp, inplace=True)
            out = F.conv2d(input=out, **params_conv1, **buffs_conv1, **self.conv1)

        params_bn2, buffs_bn2 = generate_dict_params_buffs(m, lst_buffer_names[1],
                                                           get_model_object_from_lst_buffer_name, coef, grads)
        out = F.batch_norm(input=out, training=True, **params_bn2, **buffs_bn2)
        out = F.relu(out, inplace=True)
        params_conv2, buffs_conv2 = generate_dict_params_buffs(m, lst_param_names[1],
                                                               get_model_object_from_lst_param_name, coef,
                                                               grads)
        out = F.conv2d(input=out, **params_conv2, **buffs_conv2, **self.conv2)

        if self.equalInOut:
            return torch.add(x, out)

        params_convShortcut, buffs_convShortcut = generate_dict_params_buffs(m, lst_param_names[2],
                                                                             get_model_object_from_lst_param_name, coef,
                                                                             grads)
        return torch.add(F.conv2d(input=x, **params_convShortcut, **buffs_convShortcut, **self.convShortcut), out)


class F_NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride):
        super(F_NetworkBlock, self).__init__()

        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride):
        pass
        layers = nn.ModuleList()
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1))
        return layers

    def forward(self, out, m, block_lst_param_names, block_lst_buffers_names, coef, grads):
        for layer in range(np.unique(np.array(block_lst_param_names)[:, 2]).shape[0]):
            layer_lst_params = list(filter(lambda l: l[2] == str(layer), block_lst_param_names))
            layer_lst_buffers = list(filter(lambda l: l[2] == str(layer), block_lst_buffers_names))
            out = self.layer[layer](out, m, layer_lst_params, layer_lst_buffers, coef, grads)
        return out


class F_WideResNet(nn.Module):
    def __init__(self, depth, widen_factor=1):
        super(F_WideResNet, self).__init__()

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = F_W_BasicBlock

        self.conv1 = {"stride": 1, "padding": 1, "bias": None}

        self.block1 = F_NetworkBlock(n, nChannels[0], nChannels[1], block, 1)
        self.block2 = F_NetworkBlock(n, nChannels[1], nChannels[2], block, 2)
        self.block3 = F_NetworkBlock(n, nChannels[2], nChannels[3], block, 2)

        self.nChannels = nChannels[3]

    def forward(self, x, m, coef, grads):
        lst_param_names = find_param_names(m)
        lst_buffer_names = find_buffer_names(m)

        params_conv1, buffs_conv1 = generate_dict_params_buffs(m, lst_param_names[0],
                                                               get_model_object_from_lst_param_name, coef,
                                                               grads)
        out = F.conv2d(input=x, **params_conv1, **buffs_conv1, **self.conv1)

        block1_lst_param_names = list(filter(lambda l: l[0] == "block1", lst_param_names))
        block1_lst_buffers_names = list(filter(lambda l: l[0] == "block1", lst_buffer_names))
        out = self.block1(out, m, block1_lst_param_names, block1_lst_buffers_names, coef, grads)

        block2_lst_param_names = list(filter(lambda l: l[0] == "block2", lst_param_names))
        block2_lst_buffers_names = list(filter(lambda l: l[0] == "block2", lst_buffer_names))
        out = self.block2(out, m, block2_lst_param_names, block2_lst_buffers_names, coef, grads)

        block3_lst_param_names = list(filter(lambda l: l[0] == "block3", lst_param_names))
        block3_lst_buffers_names = list(filter(lambda l: l[0] == "block3", lst_buffer_names))
        out = self.block3(out, m, block3_lst_param_names, block3_lst_buffers_names, coef, grads)

        params_bn1, buffs_bn1 = generate_dict_params_buffs(m, lst_buffer_names[-1],
                                                           get_model_object_from_lst_buffer_name, coef, grads)
        out = F.batch_norm(input=out, training=True, **params_bn1, **buffs_bn1)
        out = F.relu(out, inplace=True)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        params_linear, buffs_linear = generate_dict_params_buffs(m, lst_param_names[-1],
                                                                 get_model_object_from_lst_param_name, coef, grads)
        out = F.linear(input=out, **params_linear, **buffs_linear)
        return out


"""## Wide-Resnet-28-10"""


class W_BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super(W_BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride):
        super(NetworkBlock, self).__init__()

        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1):
        super(WideResNet, self).__init__()

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = W_BasicBlock

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def wide_resnet_28_10(num_classes=10):
    return WideResNet(depth=28,
                      num_classes=num_classes,
                      widen_factor=10)
