import json
import os

import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps

from utils import get_sdf
from scene.configs import get_bins
from scene.layout_model import LayoutNet
from scene.object_detection_model import Bdb3DNet
from scene.mesh_model import DensTMNet
from scene.relation_net import rel_cfg
from scene.utils import (get_layout_bdb_sunrgbd, get_rotation_matix_result, get_bdb_evaluation, 
    get_bdb_2d_result, get_bdb_3d_result, get_g_features, parse_detections, 
    to_dict_tensor, collate_fn, NYU40CLASSES, NYU37_TO_PIX3D_CLS_MAPPING)

HEIGHT_PATCH = 256
WIDTH_PATCH = 256
data_transforms = transforms.Compose([
    transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_data(data_path, img_path, device):
    cam_K = np.array([
        [1060.5317, 0, 951.2999],
        [0, 1060.3856, 536.7704],
        [0, 0, 1.]
    ])

    detection_path = os.path.join(data_path, 'detections.json')
    assert os.path.exists(detection_path)

    '''preprocess'''
    image = Image.open(img_path).convert('RGB')
    image_flipped = ImageOps.mirror(image)
    with open(detection_path, 'r') as file:
        detections = json.load(file)
    boxes = dict()

    bdb2D_pos, size_cls = parse_detections(detections)

    # obtain geometric features
    boxes['g_feature'] = get_g_features(bdb2D_pos, rel_cfg)

    # encode class
    cls_codes = torch.zeros([len(size_cls), len(NYU40CLASSES)])
    cls_codes[range(len(size_cls)), size_cls] = 1
    boxes['size_cls'] = cls_codes

    # get object images
    patch = []
    for bdb in bdb2D_pos:
        img = image.crop((bdb[0], bdb[1], bdb[2], bdb[3]))
        img = data_transforms(img)
        patch.append(img)
    boxes['patch'] = torch.stack(patch)
    image = data_transforms(image)
    image_flipped = data_transforms(image_flipped)
    camera = dict()
    camera['K'] = cam_K
    boxes['bdb2D_pos'] = np.array(bdb2D_pos)

    """assemble data"""
    data = collate_fn([{'image':image, 'image_flipped': image_flipped, 'boxes_batch':boxes, 'camera':camera}])
    image = data['image'].to(device)
    image_flipped = data['image_flipped'].to(device)
    K = data['camera']['K'].float().to(device)
    patch = data['boxes_batch']['patch'].to(device)
    size_cls = data['boxes_batch']['size_cls'].float().to(device)
    g_features = data['boxes_batch']['g_feature'].float().to(device)
    split = data['obj_split']
    rel_pair_counts = torch.cat([torch.tensor([0]), torch.cumsum(
        torch.pow(data['obj_split'][:, 1] - data['obj_split'][:, 0], 2), 0)], 0)
    cls_codes = torch.zeros([size_cls.size(0), 9]).to(device)
    cls_codes[range(size_cls.size(0)), [NYU37_TO_PIX3D_CLS_MAPPING[cls.item()] for cls in
                                        torch.argmax(size_cls, dim=1)]] = 1
    bdb2D_pos = data['boxes_batch']['bdb2D_pos'].float().to(device)

    input_data = {'image':image, 'image_flipped': image_flipped, 'K':K, 'patch':patch, 'g_features':g_features,
                  'size_cls':size_cls, 'split':split, 'rel_pair_counts':rel_pair_counts,
                  'cls_codes':cls_codes, 'bdb2D_pos':bdb2D_pos}
    return input_data


class SceneModel(nn.Module):
    def __init__(self, data_path, device):
        super(SceneModel, self).__init__()
        bins = get_bins()
        self.layout_estimation = nn.DataParallel(LayoutNet(bins).to(device))
        self.object_detection = Bdb3DNet(bins).to(device)
        self.mesh_reconstruction = nn.DataParallel(DensTMNet().to(device))
        self.device = device
        self.data_path = data_path # folder that contains detections.json and cam_K.txt
        self.bins_tensor = to_dict_tensor(bins, if_cuda=True)
        self.ground_criterion = nn.L1Loss()
#         self.ground_criterion = nn.MSELoss()
#         self.reg_criterion = nn.SmoothL1Loss(reduction='mean', beta=20)
        self.reg_criterion = nn.MSELoss(reduction='mean')
        self.optimizer = self.get_optimizer()

    def get_optimizer(self, lr=1e-4, betas='[0.9, 0.999]', eps=1e-8, weight_decay=1e-4):
        opt_params = []
        opt_layers = {
            'object_detection': ['fc5', 'fc_centroid', 'fc_off_1', 'fc_off_2'],
#             'layout_estimation': ['fc_5', 'fc_6']
        }
        
        self.mesh_reconstruction.train(False)
        self.object_detection.resnet.train(False)
        self.layout_estimation.module.resnet.train(False) 
        for module, layer_list in opt_layers.items():
            for layer_name in layer_list:
                if module == 'object_detection':
                    layer = getattr(getattr(self, module), layer_name)
                else:
                    layer = getattr(getattr(self, module).module, layer_name)
                for param in layer.parameters():
                    param.requires_grad = True
                    opt_params.append(param)
        print('{} parameters to optimize'.format(len(opt_params)))
        optimizer = torch.optim.Adam(opt_params,
                                     lr=lr, betas=eval(betas),
                                     eps=eps,
                                     weight_decay=weight_decay)
        return optimizer

    def load_weight(self, checkpoint_path):
        pretrained_model = torch.load(checkpoint_path)
        module_dict = {'layout_estimation': self.layout_estimation, 
                       'object_detection': self.object_detection,
                       'mesh_reconstruction': self.mesh_reconstruction}
      
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model.items() if
                           k in model_dict}
        missed_layers = set([k.split('.')[0] for k in model_dict if k not in pretrained_dict])
        if len(missed_layers) > 0:
            print(str(set(missed_layers)) + ' subnet missed.')
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
    
    def save_weight(self, checkpoint_path):
        model_dict = self.state_dict()
        torch.save(model_dict, checkpoint_path)
        
    def forward(self, img_path):
        data = load_data(self.data_path, img_path, self.device)
        pitch_reg_result, roll_reg_result, pitch_cls_result, roll_cls_result, \
            lo_ori_reg_result, lo_ori_cls_result, lo_centroid_result, lo_coeffs_result \
            = self.layout_estimation(data['image'])

        layout_output = {'pitch_reg_result':pitch_reg_result, 'roll_reg_result':roll_reg_result,
                         'pitch_cls_result':pitch_cls_result, 'roll_cls_result':roll_cls_result,
                         'lo_ori_reg_result':lo_ori_reg_result, 'lo_ori_cls_result':lo_ori_cls_result,
                         'lo_centroid_result':lo_centroid_result, 'lo_coeffs_result':lo_coeffs_result}
        size_reg_result, ori_reg_result, ori_cls_result, \
            centroid_reg_result, centroid_cls_result, \
            offset_2D_result = self.object_detection(data['patch'], data['size_cls'], data['g_features'],
                                                     data['split'], data['rel_pair_counts'])
        object_output = {'size_reg_result':size_reg_result, 'ori_reg_result':ori_reg_result,
                         'ori_cls_result':ori_cls_result, 'centroid_reg_result':centroid_reg_result,
                         'centroid_cls_result':centroid_cls_result, 'offset_2D_result':offset_2D_result}
        
        mesh_output, _, _, _, _, out_faces = self.mesh_reconstruction(data['patch'], data['cls_codes'])
        mesh_output = mesh_output[-1]
        # convert to SUNRGBD coordinates
        mesh_output[:, 2, :] *= -1
        mesh_output = {'meshes': mesh_output, 'out_faces': out_faces}

        est_data = {**layout_output, **object_output, **mesh_output}

        lo_bdb3D_out = get_layout_bdb_sunrgbd(
            self.bins_tensor, est_data['lo_ori_reg_result'],
            torch.argmax(est_data['lo_ori_cls_result'], 1),
            est_data['lo_centroid_result'], est_data['lo_coeffs_result'])

        cam_R_out = get_rotation_matix_result(
            self.bins_tensor, torch.argmax(est_data['pitch_cls_result'], 1),
            est_data['pitch_reg_result'],
            torch.argmax(est_data['roll_cls_result'], 1), est_data['roll_reg_result'])

        # projected center
        P_result = torch.stack((
            (data['bdb2D_pos'][:, 0] + data['bdb2D_pos'][:, 2]) / 2 - (data['bdb2D_pos'][:, 2] - data['bdb2D_pos'][:, 0]) * est_data['offset_2D_result'][:,0],
            (data['bdb2D_pos'][:, 1] + data['bdb2D_pos'][:, 3]) / 2 - (data['bdb2D_pos'][:, 3] - data['bdb2D_pos'][:, 1]) * est_data['offset_2D_result'][:, 1]),
            1)
        
        # parse box results
        bdb3D_out_form_cpu, bdb3D_out_form, bdb3D_out = get_bdb_evaluation(
            self.bins_tensor, torch.argmax(est_data['ori_cls_result'], 1),
            est_data['ori_reg_result'], torch.argmax(est_data['centroid_cls_result'], 1),
            est_data['centroid_reg_result'], data['size_cls'], est_data['size_reg_result'], P_result,
            data['K'], cam_R_out, data['split'], return_bdb=True)

        # get 2D boxes in pixel coordinates
        bdb2D_result = get_bdb_2d_result(bdb3D_out, cam_R_out, data['K'], data['split'])
        bdb2D_result[:, 0] *= data['K'][0,0,2].item(); bdb2D_result[:, 1] *= data['K'][0,1,2].item()
        bdb2D_result[:, 2] *= data['K'][0,0,2].item(); bdb2D_result[:, 3] *= data['K'][0,1,2].item()
        
        mesh_verts = est_data['meshes'].transpose(1, 2)
        mesh_faces = est_data['out_faces']
        
        # bdb3D_out: (N, 3), 
        return lo_bdb3D_out, bdb3D_out_form, bdb3D_out_form_cpu, bdb3D_out, \
            bdb2D_result, data['bdb2D_pos'], \
            cam_R_out, mesh_verts, mesh_faces
    
    def inference(self, image_path):
        with torch.no_grad():
            out = self.forward(image_path)
        lo_bdb3D_out, bdb3D_out_form, bdb3D_out_form_cpu, bdb3D_out, \
        bdb2D_result, bdb2D_input, cam_R_out, mesh_verts, mesh_faces = out
        verts, faces = self.get_scene_mesh(mesh_verts, mesh_faces, bdb3D_out_form, cam_R=cam_R_out[0])
        verts[:, 2] = - verts[:, 2]
        verts[:, 1] = - verts[:, 1]
        verts = verts[:, [2, 1, 0]]
        cam_R = cam_R_out[0].detach().cpu().numpy()
        lo_bdb3D_world_coord = lo_bdb3D_out[0].detach().cpu().numpy().copy()
        lo_bdb3D_cam_coord = lo_bdb3D_out[0].detach().cpu().numpy().copy().dot(cam_R)
        # convert to coordinate system used for the body
#         lo_bdb3D_cam_coord[:, 2] = - lo_bdb3D_cam_coord[:, 2]
        lo_bdb3D_cam_coord[:, 1] = - lo_bdb3D_cam_coord[:, 1]
        lo_bdb3D_cam_coord = lo_bdb3D_cam_coord[:, [2, 1, 0]]
        return verts, faces, cam_R, lo_bdb3D_world_coord, lo_bdb3D_cam_coord
        
    def calc_scene_loss(self, image_path, **kwargs):
        lo_bdb3D_out, bdb3D_out_form, bdb3D_out_form_cpu, bdb3D_out, \
            bdb2D_result, bdb2D_input, cam_R_out, mesh_verts, mesh_faces = self.forward(image_path)
        
        nobj = bdb3D_out.shape[0]
        obj_ground_weight = kwargs.get('object_groud_weight', 1)
        sdf_pen_weight = kwargs.get('sdf_pen_weight', 1)
        
        ############# Compute 2D reprojection Loss ##############
        bdb2D_loss = self.reg_criterion(bdb2D_result, bdb2D_input)
        
        ############# Compute Ground Loss ##############
        obj_grd_loss = torch.tensor(0.).to(self.device)
        if obj_ground_weight > 0.:
            ground_y = lo_bdb3D_out[0, :, 1].min()
            for obj_i in range(nobj):
                obj_grd_loss += self.ground_criterion(bdb3D_out[obj_i, :, 1].min(), ground_y)
                
        ############# Compute penetration Loss ##############
        obj_pen_loss = torch.tensor(0.).to(self.device)
        if sdf_pen_weight > 0. and nobj > 1:
            grid_dim = kwargs.get('sdf_grid_dim', 8)
            for i_ in range(nobj):
                item_verts = self.get_mesh_vertices(mesh_verts[i_], bdb3D_out_form[i_])
                sub_idx = [j for j in range(nobj) if j!=i_]
                
                sub_scene_verts, sub_scene_faces = self.get_scene_mesh(
                        mesh_verts[sub_idx], mesh_faces[sub_idx], [bdb3D_out_form[j] for j in sub_idx])

                vmin = sub_scene_verts.min(0)
                vmax = sub_scene_verts.max(0)
                
                qp_sdfs = get_sdf(sub_scene_verts, sub_scene_faces, grid_dim, vmin, vmax).to(self.device)
                norm_verts = (item_verts-sub_scene_verts.min(0)[0])/(sub_scene_verts.max(0)[0]-sub_scene_verts.min(0)[0])*2-1
                item_sdf = F.grid_sample(
                         qp_sdfs.reshape(1, 1, grid_dim, grid_dim, grid_dim).float(),
                         norm_verts.reshape(1, -1, 1, 1, 3), padding_mode='border')
                curr_pen_loss = (item_sdf[item_sdf < 0].unsqueeze(dim=-1).abs()).pow(2).sum(dim=-1).sqrt().sum()
                obj_pen_loss += curr_pen_loss
                
        obj_grd_loss *= obj_ground_weight
        obj_pen_loss *= sdf_pen_weight
        total_loss = obj_grd_loss + obj_pen_loss + bdb2D_loss
        return {'obj_grd_loss': obj_grd_loss, 'obj_pen_loss': obj_pen_loss, 'reproj_loss': bdb2D_loss}
    
    def optimize(self, is_joint, iters, **kwargs):
        loss_func = self.calc_joint_loss if is_joint else self.calc_scene_loss
        loss_hist = []
        for _ in range(iters):
            losses = loss_func(**kwargs)
            loss_sum = sum(list(losses.values()))
            self.optimizer.zero_grad()
            loss_sum.backward()
            self.optimizer.step()
            loss_hist.append(losses)
            print(losses, loss_sum)
        return loss_hist
    
    def calc_joint_loss(self, image_path, body_verts, body_faces, **kwargs):
        lo_bdb3D_out, bdb3D_out_form, bdb3D_out_form_cpu, bdb3D_out, \
            bdb2D_result, bdb2D_input, cam_R_out, mesh_verts, mesh_faces = self.forward(image_path)
        nobj = bdb3D_out.shape[0]
        object_groud_weight = kwargs.get('object_groud_weight', 10)
        sdf_pen_weight = kwargs.get('sdf_pen_weight', 0.1)
        body_ground_weight = kwargs.get('body_ground_weight', 10)
        mesh_verts, mesh_faces = self.get_scene_mesh(mesh_verts, mesh_faces, bdb3D_out_form, to_numpy=False, cam_R=cam_R_out[0])
        
        body_verts = body_verts[0]
        body_verts_cam_coord = body_verts.dot(np.linalg.inv(cam_R_out[0].detach().cpu().numpy()))
        
        ############# Compute 2D reprojection Loss ##############
        bdb2D_loss = self.reg_criterion(bdb2D_result, bdb2D_input)
        
        ############# Compute Ground Loss ##############
        obj_grd_loss = torch.tensor(0.).to(self.device)
        if object_groud_weight > 0.:
            ground_y = lo_bdb3D_out[0, :, 1].min()
            for obj_i in range(nobj):
                obj_grd_loss += self.ground_criterion(bdb3D_out[obj_i, :, 1].min(), ground_y)
                
        body_grd_loss = torch.tensor(0.).to(self.device)
        if body_ground_weight > 0.:
            ground_y = lo_bdb3D_out[0, :, 1].min()
            body_min = torch.tensor(body_verts_cam_coord[:, 1].min()).to(self.device)
            body_grd_loss += self.ground_criterion(body_min, ground_y)
            
        ############# Compute penetration Loss ##############
        obj_pen_loss = torch.tensor(0.).to(self.device)
        if sdf_pen_weight > 0.:
            grid_dim = kwargs.get('sdf_grid_dim', 8)
            vmin = body_verts.min(0)
            vmax = body_verts.max(0)
            vmin_tensor = torch.tensor(body_verts.min(0)).to(self.device)
            vmax_tensor = torch.tensor(body_verts.max(0)).to(self.device)
            qp_sdfs = get_sdf(body_verts, body_faces, grid_dim, vmin, vmax).to(self.device)
            norm_verts = (mesh_verts-vmin_tensor)/(vmax_tensor-vmin_tensor)*2-1
            item_sdf = F.grid_sample(
                     qp_sdfs.reshape(1, 1, grid_dim, grid_dim, grid_dim).float(),
                     norm_verts.reshape(1, -1, 1, 1, 3), padding_mode='border')
            curr_pen_loss = (item_sdf[item_sdf < 0].unsqueeze(dim=-1).abs()).pow(2).sum(dim=-1).sqrt().sum()
            obj_pen_loss += curr_pen_loss
        
        obj_grd_loss *= object_groud_weight
        body_grd_loss *= body_ground_weight
        obj_pen_loss *= sdf_pen_weight
        total_loss = obj_grd_loss + body_grd_loss + obj_pen_loss + bdb2D_loss
        return {'obj_grd_loss': obj_grd_loss, 'body_grd_loss': body_grd_loss, 
                'obj_pen_loss': obj_pen_loss, 'reproj_loss': bdb2D_loss}
    
    def get_mesh_vertices(self, mesh_vertices, bbox, cam_R=None):
        "Scale and translate unit-sized mesh by box properties."
        # mesh_vertices (V, 3).
        mesh_center = (mesh_vertices.max(0)[0] + mesh_vertices.min(0)[0]) / 2.
        mesh_vertices = mesh_vertices - mesh_center
        mesh_coef = (mesh_vertices.max(0)[0] - mesh_vertices.min(0)[0]) / 2.
        if isinstance(bbox['coeffs'], torch.Tensor):
            bbox_coeffs = bbox['coeffs'].to(self.device)
            bbox_centroid = bbox['centroid'].to(self.device)
            bbox_basis = bbox['basis'].to(self.device)
        else:
            bbox_coeffs = torch.tensor(bbox['coeffs']).to(self.device)
            bbox_centroid = torch.tensor(bbox['centroid']).to(self.device)
            bbox_basis = torch.tensor(bbox['basis']).to(self.device)
        
        mesh_vertices = torch.matmul(mesh_vertices, torch.diag(1./mesh_coef).to(self.device))
        mesh_vertices = torch.matmul(mesh_vertices, torch.diag(bbox_coeffs))

        # set orientation
        mesh_vertices = torch.matmul(mesh_vertices, bbox_basis)

        # move to center
        mesh_vertices = mesh_vertices + bbox_centroid
        if cam_R is not None:
            mesh_vertices = torch.matmul(mesh_vertices, cam_R)
        return mesh_vertices
    
    def get_scene_mesh(self, mesh_vertices, mesh_faces, box_params, 
                       flatten=True, to_numpy=True, cam_R=None):
        N, V = mesh_vertices.shape[0], mesh_vertices.shape[1]
        scene_mesh_vertices = []
        scene_mesh_faces = []
        for i in range(N):
            mesh_v = self.get_mesh_vertices(mesh_vertices[i], box_params[i], cam_R)
            scene_mesh_vertices.append(mesh_v)
            if not flatten:
                scene_mesh_faces.append(mesh_faces[i]-1)
            else:
                scene_mesh_faces.append(mesh_faces[i] + i*V - 1)
        if flatten:
            scene_mesh_vertices = torch.cat(scene_mesh_vertices, dim=0)
            scene_mesh_faces = torch.cat(scene_mesh_faces, dim=0)
            
        if to_numpy:
            scene_mesh_vertices = scene_mesh_vertices.detach().cpu().numpy()
            scene_mesh_faces = scene_mesh_faces.detach().cpu().numpy()
        return scene_mesh_vertices, scene_mesh_faces
    
