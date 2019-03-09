import os
import sys
import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class MyDenseNet(nn.Module):
    def __init__(self, depth, num_classes, pretrained=True):
        super(MyDenseNet, self).__init__()
        if depth == 121:
            model = models.densenet121(pretrained)
        elif depth == 169:
            model = models.densenet169(pretrained)
        elif depth == 201:
            model = models.densenet201(pretrained)
        elif depth == 161:
            model = models.densenet161(pretrained)

        self.num_ftrs = model.classifier.in_features
        # self.num_classes = num_classes

        self.shared = nn.Sequential(*list(model.children())[:-1])
        self.target = self.build_classifier(self.num_ftrs, [128, 128], num_classes)
        # self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes))

    def forward(self, x):
        # pdb.set_trace()

        x = self.shared(x)
        x = torch.squeeze(x)
        return self.target(x)

    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive'%to_layer)

        # if to_layer = -1, frozen all
        child_counter = 0
        for child in self.shared.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1

    @staticmethod
    def build_classifier(num_in_features, hidden_layers, num_out_features):

        classifier = nn.Sequential()
        if hidden_layers == None:
            classifier.add_module('fc0', nn.Linear(num_in_features, 102))
        else:
            layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
            classifier.add_module('fc0', nn.Linear(num_in_features, hidden_layers[0]))
            classifier.add_module('relu0', nn.ReLU())
            classifier.add_module('drop0', nn.Dropout(0.5))
            for i, (h1, h2) in enumerate(layer_sizes):
                classifier.add_module('fc' + str(i + 1), nn.Linear(h1, h2))
                classifier.add_module('relu' + str(i + 1), nn.ReLU())
                classifier.add_module('drop' + str(i + 1), nn.Dropout(.5))
            classifier.add_module('output', nn.Linear(hidden_layers[-1], num_out_features))

        return classifier


class MyResNet(nn.Module):
    def __init__(self, depth, num_classes, pretrained=True):
        super(MyResNet, self).__init__()
        if depth == 18:
            model = models.resnet18(pretrained)
        elif depth == 34:
            model = models.resnet34(pretrained)
        elif depth == 50:
            model = models.resnet50(pretrained)
        elif depth == 152:
            model = models.resnet152(pretrained)

        self.num_ftrs = model.fc.in_features
        # self.num_classes = num_classes

        self.shared = nn.Sequential(*list(model.children())[:-1])
        self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes))

    def forward(self, x):
        # pdb.set_trace()

        x = self.shared(x)
        x = torch.squeeze(x)
        return self.target(x)

    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive'%to_layer)

        # if to_layer = -1, frozen all
        child_counter = 0
        for child in self.shared.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1


# def test():
#     net = Net2(num_classes=43)
#     y = net(torch.randn(1, 1, 32, 32))
#     print(y.size())


# test()
