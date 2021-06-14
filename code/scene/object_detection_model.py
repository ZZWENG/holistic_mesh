import torch
import torch.nn as nn
from .resnet import resnet34
from .relation_net import RelationNet

NYU40CLASSES = 41


class Bdb3DNet(nn.Module):
    def __init__(self, bins):
        super(Bdb3DNet, self).__init__()
        '''Module parameters'''
        self.OBJ_ORI_BIN = len(bins['ori_bin'])
        self.OBJ_CENTER_BIN = len(bins['centroid_bin'])

        # set up neural network blocks
        resnet = nn.DataParallel(resnet34(pretrained=False))
        self.resnet = resnet

        # set up relational network blocks
        self.relnet = RelationNet()

        # branch to predict the size
        self.fc1 = nn.Linear(2048 + NYU40CLASSES, 128)
        self.fc2 = nn.Linear(128, 3)

        # branch to predict the orientation
        self.fc3 = nn.Linear(2048 + NYU40CLASSES, 128)
        self.fc4 = nn.Linear(128, self.OBJ_ORI_BIN * 2)

        # branch to predict the centroid
        self.fc5 = nn.Linear(2048 + NYU40CLASSES, 128)
        self.fc_centroid = nn.Linear(128, self.OBJ_CENTER_BIN * 2)

        # branch to predict the 2D offset
        self.fc_off_1 = nn.Linear(2048 + NYU40CLASSES, 128)
        self.fc_off_2 = nn.Linear(128, 2)

        self.relu_1 = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(p=0.5)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

    def forward(self, x, size_cls, g_features, split, rel_pair_counts):
        # get appearance feature from resnet.
        a_features = self.resnet(x)
        a_features = a_features.view(a_features.size(0), -1)

        # extract relational features from other objects.
        r_features = self.relnet(a_features, g_features, split, rel_pair_counts)

        a_r_features = torch.add(a_features, r_features)

        # add object category information
        a_r_features = torch.cat([a_r_features, size_cls], 1)

        # branch to predict the size
        size = self.fc1(a_r_features)
        size = self.relu_1(size)
        size = self.dropout_1(size)
        size = self.fc2(size)

        # branch to predict the orientation
        ori = self.fc3(a_r_features)
        ori = self.relu_1(ori)
        ori = self.dropout_1(ori)
        ori = self.fc4(ori)
        ori = ori.view(-1, self.OBJ_ORI_BIN, 2)
        ori_reg = ori[:, :, 0]
        ori_cls = ori[:, :, 1]

        # branch to predict the centroid
        centroid = self.fc5(a_r_features)
        centroid = self.relu_1(centroid)
        centroid = self.dropout_1(centroid)
        centroid = self.fc_centroid(centroid)
        centroid = centroid.view(-1, self.OBJ_CENTER_BIN, 2)
        centroid_cls = centroid[:, :, 0]
        centroid_reg = centroid[:, :, 1]

        # branch to predict the 2D offset
        offset = self.fc_off_1(a_r_features)
        offset = self.relu_1(offset)
        offset = self.dropout_1(offset)
        offset = self.fc_off_2(offset)

        return size, ori_reg, ori_cls, centroid_reg, centroid_cls, offset
