import os, pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet18_full


def load_template():
    curr_path = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(curr_path, 'data', 'sphere2562.pkl')

    with open(file_name, 'rb') as file:
        sphere_obj = pickle.load(file)
        sphere_points_normals = torch.from_numpy(sphere_obj['v']).float()
        sphere_faces = torch.from_numpy(sphere_obj['f']).long()
        sphere_adjacency = torch.from_numpy(sphere_obj['adjacency'].todense()).long()
        sphere_edges = torch.from_numpy(sphere_obj['edges']).long()
        sphere_edge2face = torch.from_numpy(sphere_obj['edge2face'].todense()).type(torch.uint8)
    return sphere_points_normals, sphere_faces, sphere_adjacency, sphere_edges, sphere_edge2face


sphere_points_normals, sphere_faces, sphere_adjacency, sphere_edges, sphere_edge2face = load_template()


def sample_points_on_edges(points, edges, quantity = 1, mode = 'train'):
    n_batch = edges.shape[0]
    n_edges = edges.shape[1]

    if mode == 'train':
        # if the sampling rate is larger than 1, we randomly pick points on faces.
        weights = np.diff(np.sort(np.vstack(
            [np.zeros((1, n_edges * quantity)), np.random.uniform(0, 1, size=(1, n_edges * quantity)),
             np.ones((1, n_edges * quantity))]), axis=0), axis=0)
    else:
        # if in test mode, we pick the central point on faces.
        weights = 0.5 * np.ones((2, n_edges * quantity))

    weights = weights.reshape([2, quantity, n_edges])
    weights = torch.from_numpy(weights).float().to(points.device)
    weights = weights.transpose(1, 2)
    weights = weights.transpose(0, 1).contiguous()
    weights = weights.expand(n_batch, n_edges, 2, quantity).contiguous()
    weights = weights.view(n_batch * n_edges, 2, quantity)

    left_nodes = torch.gather(points.transpose(1, 2), 1,
                              (edges[:, :, 0] - 1).unsqueeze(-1).expand(edges.size(0), edges.size(1), 3))
    right_nodes = torch.gather(points.transpose(1, 2), 1,
                              (edges[:, :, 1] - 1).unsqueeze(-1).expand(edges.size(0), edges.size(1), 3))

    edge_points = torch.cat([left_nodes.unsqueeze(-1), right_nodes.unsqueeze(-1)], -1).view(n_batch*n_edges, 3, 2)

    new_point_set = torch.bmm(edge_points, weights).contiguous()
    new_point_set = new_point_set.view(n_batch, n_edges, 3, quantity)
    new_point_set = new_point_set.transpose(2, 3).contiguous()
    new_point_set = new_point_set.view(n_batch, n_edges * quantity, 3)
    new_point_set = new_point_set.transpose(1, 2).contiguous()
    return new_point_set


#initialize the weighs of the network for Convolutional layers and batchnorm layers
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
        
class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 2500, output_dim = 3):
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck_size, bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(bottleneck_size, bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(bottleneck_size//2, bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(bottleneck_size//4, output_dim, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size//4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class EREstimate(nn.Module):
    def __init__(self, bottleneck_size=2500, output_dim = 3):
        super(EREstimate, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck_size, bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(bottleneck_size, bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(bottleneck_size//2, bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(bottleneck_size//4, output_dim, 1)

        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size//4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x


class DensTMNet(nn.Module):
    def __init__(self, bottleneck_size=1024, n_classes=9):
        super(DensTMNet, self).__init__()
        self.num_points = 2562
        self.subnetworks = 2
        self.training = False
        self.train_e_e = True

        self.encoder = resnet18_full(pretrained=False, num_classes=1024)
        self.decoders = nn.ModuleList(
            [PointGenCon(bottleneck_size=3 + bottleneck_size + n_classes) for i in range(0, self.subnetworks)])

        if self.train_e_e:
            self.error_estimators = nn.ModuleList(
                [EREstimate(bottleneck_size=3 + bottleneck_size + n_classes, output_dim=1) for i in range(0, max(self.subnetworks-1, 1))])
            self.face_samples = 1

        # initialize weight
        self.apply(weights_init)

    def forward(self, image, size_cls, threshold = 0.1, factor = 1.):
        mode = 'train' if self.training else 'test'
        device = image.device
        n_batch = image.size(0)
        n_edges = sphere_edges.shape[0]

        if mode == 'test':
            current_faces = sphere_faces.clone().unsqueeze(0).to(device)
            current_faces = current_faces.repeat(n_batch, 1, 1)
        else:
            current_faces = None

        current_edges = sphere_edges.clone().unsqueeze(0).to(device)
        current_edges = current_edges.repeat(n_batch, 1, 1)

        # image encoding
        image = image[:,:3,:,:].contiguous()
        image = self.encoder(image)
        image = torch.cat([image, size_cls], 1)

        current_shape_grid = sphere_points_normals[:, :3].t().expand(n_batch, 3, self.num_points).to(device)

        # outputs for saving
        out_shape_points = []
        out_sampled_mesh_points = []
        out_indicators = []

        # boundary faces for boundary refinement
        boundary_point_ids = torch.zeros(size=(n_batch, self.num_points), dtype=torch.uint8).to(device)
        remove_edges_list = []

        # AtlasNet deformation + topoly modification
        for i in range(self.subnetworks):
            current_image_grid = image.unsqueeze(2).expand(image.size(0), image.size(1),
                                                           current_shape_grid.size(2)).contiguous()
            current_image_grid = torch.cat((current_shape_grid, current_image_grid), 1).contiguous()
            current_shape_grid = current_shape_grid + self.decoders[i](current_image_grid)

            # save deformed point cloud
            out_shape_points.append(current_shape_grid)

            if i == self.subnetworks - 1 and self.subnetworks > 1:
                remove_edges_list = [item for item in remove_edges_list if len(item)]
                if remove_edges_list:
                    remove_edges_list = torch.unique(torch.cat(remove_edges_list), dim=0)
                    for batch_id in range(n_batch):
                        rm_edges = remove_edges_list[remove_edges_list[:, 0] == batch_id, 1]
                        if len(rm_edges) > 0:
                            rm_candidates, counts = torch.unique(sphere_edges[rm_edges], return_counts=True)
                            boundary_ids = counts < sphere_adjacency[rm_candidates - 1].sum(1)
                            boundary_point_ids[batch_id][rm_candidates[boundary_ids] - 1] = 1

                return out_shape_points, out_sampled_mesh_points, out_indicators, current_edges, boundary_point_ids, current_faces

            if self.train_e_e:
                # sampling from deformed mesh
                sampled_points = sample_points_on_edges(current_shape_grid, current_edges, quantity=self.face_samples, mode=mode)

                # save sampled points from deformed mesh
                out_sampled_mesh_points.append(sampled_points)

                # preprare for face error estimation
                current_image_grid = image.unsqueeze(2).expand(image.size(0), image.size(1), sampled_points.size(2)).contiguous()
                current_image_grid = torch.cat((sampled_points, current_image_grid), 1).contiguous()

                # estimate the distance from deformed points to gt mesh.
                indicators = self.error_estimators[i](current_image_grid)
                indicators = indicators.view(n_batch, 1, n_edges, self.face_samples)
                indicators = indicators.squeeze(1)
                indicators = torch.mean(indicators, dim=2)

                # save estimated distance values from deformed points to gt mesh.
                out_indicators.append(indicators)
                # remove faces and modify the topology
                remove_edges = torch.nonzero(torch.sigmoid(indicators) < threshold)
                remove_edges_list.append(remove_edges)

                for batch_id in range(n_batch):
                    rm_edges = remove_edges[remove_edges[:, 0] == batch_id, 1]
                    if len(rm_edges)>0:
                        # cutting edges in training
                        current_edges[batch_id][rm_edges, :] = 1
                        if mode == 'test':
                            current_faces[batch_id][sphere_edge2face[rm_edges].sum(0).type(torch.bool), :] = 1

                threshold *= factor

        return out_shape_points, out_sampled_mesh_points, out_indicators, current_edges, boundary_point_ids, current_faces
