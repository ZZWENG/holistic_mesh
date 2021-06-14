import os.path as osp
from collections import defaultdict
from tqdm import tqdm
import cv2, json, time, os
import pickle

# third party imports
import numpy as np
import torch
import smplx
from psbody.mesh import Mesh
import scipy.sparse as sparse

from .vposer import VPoserDecoder
from .optimizers import optim_factory
from . import fitting
from .misc_utils import smpl_to_openpose, JointMapper
from .camera import create_camera
from .prior import create_prior, DEFAULT_DTYPE


class BodyModel(object):
    def __init__(self, args):
        for key in args:
            setattr(self, key, args[key])
       
        # Create the functions or classes for calculating priors
        self.dtype = DEFAULT_DTYPE
        self.device = torch.device('cuda')
        self.body_pose_prior = create_prior(prior_type=self.body_prior_type, **args).to(device=self.device)
        self.jaw_prior = create_prior(prior_type=self.jaw_prior_type, **args).to(device=self.device)
        self.left_hand_prior = create_prior(prior_type=self.left_hand_prior_type, **args).to(device=self.device)
        self.right_hand_prior = create_prior(prior_type=self.right_hand_prior_type, **args).to(device=self.device)
        self.expr_prior = create_prior(prior_type='l2', **args).to(device=self.device)
        self.shape_prior = create_prior(prior_type='l2', **args).to(device=self.device)
        self.angle_prior = create_prior(prior_type='angle', dtype=self.dtype).to(device=self.device)

        self.model_args = args
        self.jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')), self.jaw_pose_prior_weights)
        self.jaw_pose_prior_weights = [list(w) for w in self.jaw_pose_prior_weights]

        print('Loading vposer ckpt from', self.model_args['vposer_ckpt'])
        vposer = VPoserDecoder(latent_dim=32, dtype=self.dtype, **self.model_args)
        vposer = vposer.to(device=self.device)
        vposer.eval()
        self.vposer = vposer
        
        if self.interpenetration:
            # parse the args needed to interpenetration calculation
            self.interpenetration_args = self.get_search_tree()

        opt_weights_dict = {
            'data_weight': self.data_weights,
            'body_pose_weight': self.body_pose_prior_weights,
            'shape_weight': self.shape_weights,
            'face_weight': self.face_joints_weights,
            'expr_prior_weight': self.expr_weights,
            'sdf_penetration_weight': self.sdf_penetration_weights,
            'contact_loss_weight': self.contact_loss_weights,
            'coll_loss_weight': self.coll_loss_weights,
            'hand_prior_weight': self.hand_pose_prior_weights,
            'hand_weight': self.hand_joints_weights,
            'jaw_prior_weight': self.jaw_pose_prior_weights
        }
        keys = opt_weights_dict.keys()
        opt_weights = [dict(zip(keys, vals)) for vals in zip(*(opt_weights_dict[k] for k in keys
                       if opt_weights_dict[k] is not None))]
        for weight_list in opt_weights:
            for key in weight_list:
                weight_list[key] = torch.tensor(weight_list[key], device=self.device, dtype=self.dtype)

        self.opt_weights = opt_weights  # list of dict.
        self.init_joints_idxs = torch.tensor(args['init_joints_idxs'], device=self.device)
        self.bm = self.create_single_body(create_body_pose=True)
    
    def create_camera(self):
        camera_center = torch.tensor([self.camera_center_x, self.camera_center_y], dtype=self.dtype).view(-1, 2)
        camera = create_camera(focal_length_x=self.focal_length_x, focal_length_y=self.focal_length_y,
                               center=camera_center, dtype=self.dtype).to(device=self.device)
        if hasattr(camera, 'rotation'):
            camera.rotation.requires_grad = False
        return camera
 
    def create_single_body(self, create_body_pose=False):
        joint_mapper = JointMapper(smpl_to_openpose(self.model_type, use_hands=self.use_hands, use_face=self.use_face))
        model_params = dict(model_path=self.model_folder,
                            joint_mapper=joint_mapper, create_body_pose=create_body_pose, dtype=self.dtype, **self.model_args)
        body_model = smplx.create(**model_params)
        body_model.reset_params()
        body_model.transl.requires_grad = True
        body_model.to(device=self.device)
        return body_model
    
    def create_pose_embeddings(self):
        return torch.zeros([1, 32], dtype=self.dtype, device=self.device, requires_grad=True)

    def get_joints_weights(self, use_face, use_hands):
        NUM_BODY_JOINTS = 25
        NUM_HAND_JOINTS = 20
        num_joints = (NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS * use_hands)
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(num_joints + 2 * use_hands + use_face * 51, dtype=np.float32)
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)
    
    def get_search_tree(self):
        # Create the search tree for interpenetration computation
        search_tree = None
        pen_distance = None
        filter_faces = None
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces
        df_cone_height = 0.5
        search_tree = BVH(max_collisions=8)
        pen_distance = collisions_loss.DistanceFieldPenetrationLoss(
                sigma=df_cone_height, point2plane=False,
                vectorized=True, penalize_outside=True)
        if self.model_args["part_segm_fn"]:
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(self.part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file, encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents,
                ign_part_pairs=self.ign_part_pairs).to(device=self.device)
        return search_tree, pen_distance, filter_faces

    def fit(self, batch, maxiter, scene_verts=None, scene_faces=None,
            sdf=None, vmin=None, vmax=None, grid_dim=None, cam_R=None, ground_y=None):
        keypoints = batch['keypoints']
        args = self.model_args
        self.results = [dict() for _ in range(len(keypoints))]
        bm = self.create_single_body(len(keypoints))
        
        if self.sdf_penetration and scene_verts is not None:
            assert sdf is not None
            sdf_penetration = True
        else:
            sdf_penetration = False
            
        if self.contact and scene_verts is not None:
            contact = True
            contact_verts_ids = []
            body_segments_dir = self.model_args['body_segments_dir']
            for part in self.model_args['contact_body_parts']:
                with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
                    data = json.load(f)
                    contact_verts_ids.append(list(set(data["verts_ind"])))
            contact_verts_ids = np.concatenate(contact_verts_ids)
            vertices = bm(return_verts=True, body_pose= torch.zeros((len(keypoints), 63), dtype=self.dtype, device=self.device)).vertices
            vertices_np = vertices.detach().cpu().numpy().squeeze()
            body_faces_np = bm.faces_tensor.detach().cpu().numpy().reshape(-1, 3)
            m = Mesh(v=vertices_np, f=body_faces_np)
            ftov = m.faces_by_vertex(as_sparse_matrix=True)
            ftov = sparse.coo_matrix(ftov)
            indices = torch.LongTensor(np.vstack((ftov.row, ftov.col))).to(self.device)
            values = torch.FloatTensor(ftov.data).to(self.device)
            ftov = torch.sparse.FloatTensor(indices, values, torch.Size(ftov.shape))
            scene = Mesh(v=scene_verts, f=scene_faces)
            scene.vn = scene.estimate_vertex_normals()
            scene_v = torch.tensor(scene.v[np.newaxis, :], dtype=self.dtype, device=self.device).contiguous()
            scene_vn = torch.tensor(scene.vn[np.newaxis, :], dtype=self.dtype, device=self.device)
            scene_f = torch.tensor(scene.f.astype(int)[np.newaxis, :], dtype=torch.long, device=self.device)
        else:
            contact = False
            contact_verts_ids = ftov =scene_v = scene_f = scene_vn = None
            
        pose_embeddings = self.create_pose_embeddings() 
        gt_joints = keypoints[:,:,:2].to(device=self.device, dtype=self.dtype)
        joint_weights = self.get_joints_weights(self.use_hands, self.use_face).to(device=self.device).unsqueeze_(dim=0)
        camera = self.create_camera()
        if self.use_joints_conf:
            joints_conf = keypoints[:, :, 2].to(device=self.device, dtype=self.dtype)
        init_t = fitting.guess_init(bm, gt_joints, self.body_tri_idxs,
                                    use_vposer=True, vposer=self.vposer,
                                    pose_embedding=pose_embeddings, model_type=self.model_type,
                                    focal_length=self.focal_length_x, dtype=self.dtype)
        camera_loss = fitting.create_loss('camera_init', trans_estimation=init_t,
                                          init_joints_idxs=self.init_joints_idxs,
                                          camera_mode=self.camera_mode, dtype=self.dtype).to(device=self.device)
        camera_loss.trans_estimation[:] = init_t
        
        if self.interpenetration:
            search_tree = self.interpenetration_args[0]
            pen_distance = self.interpenetration_args[0]
            filter_faces = self.interpenetration_args[0]
        else:
            search_tree = pen_distance = filter_faces = None
            
        loss = fitting.create_loss(loss_type=self.loss_type,
                                   joint_weights=joint_weights, rho=self.rho,
                                   use_joints_conf=self.use_joints_conf,
                                   use_face=self.use_face, use_hands=self.use_hands,
                                   vposer=self.vposer, pose_embedding=pose_embeddings,
                                   body_pose_prior=self.body_pose_prior, shape_prior=self.shape_prior,
                                   angle_prior=self.angle_prior, expr_prior=self.expr_prior,
                                   left_hand_prior=self.left_hand_prior, right_hand_prior=self.right_hand_prior,
                                   jaw_prior=self.jaw_prior,
                                   interpenetration=self.interpenetration,
                                   pen_distance=pen_distance, search_tree=search_tree, tri_filtering_module=filter_faces,
                                   sdf_penetration=sdf_penetration,
                                   voxel_size=grid_dim, grid_min=vmin, grid_max=vmax, sdf=sdf,
                                   contact=contact, contact_verts_ids=contact_verts_ids,
                                   rho_contact=self.rho_contact, contact_angle=self.contact_angle,
                                   dtype=self.dtype)
        loss = loss.to(device=self.device)
        latent_mean  = torch.zeros([len(keypoints), 32], device=self.device, requires_grad=True, dtype=self.dtype)        
        body_mean_pose = self.vposer(latent_mean).detach().cpu()

        with fitting.FittingMonitor(model_type=self.model_type, maxiters=maxiter) as monitor:
            # Reset the parameters to estimate the initial translation of the body model
            assert(self.camera_mode == 'fixed')
            bm.reset_params(body_pose=body_mean_pose, transl=init_t)
            camera_opt_params = [bm.transl, bm.global_orient]
            shoulder_dist = torch.dist(gt_joints[:, 2], gt_joints[:, 5])
            try_both_orient = shoulder_dist.item() < self.side_view_thsh
            camera_optimizer, camera_create_graph = optim_factory.create_optimizer(camera_opt_params, **args)

            # The closure passed to the optimizer
            fit_camera = monitor.create_fitting_closure(
                camera_optimizer, bm, camera, gt_joints,
                camera_loss, create_graph=camera_create_graph,
                use_vposer=True, vposer=self.vposer,
                pose_embedding=pose_embeddings, return_full_pose=False, return_verts=False)

            # Step 1: Optimize over the torso joints the camera translation
            # Initialize the computational graph by feeding the initial translation
            # of the camera and the initial pose of the body model.
            camera_init_start = time.time()
            cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                                    fit_camera, camera_opt_params, bm,
                                                    use_vposer=True, pose_embedding=pose_embeddings,
                                                    vposer=self.vposer)            
            if self.interactive:
                torch.cuda.synchronize()
                tqdm.write('Camera initialization done after {:.4f}'.format(
                    time.time() - camera_init_start))
                tqdm.write('Camera initialization final loss {:.4f}'.format(
                    cam_init_loss_val))

            if try_both_orient:
                body_orient = bm.global_orient.detach().cpu().numpy()
                flipped_orient = cv2.Rodrigues(body_orient)[0].dot(cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
                flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()
                flipped_orient = torch.tensor(flipped_orient, dtype=self.dtype, device=self.device).unsqueeze(dim=0)
                orientations = [body_orient, flipped_orient]
            else:
                orientations = [bm.global_orient.detach().cpu().numpy()]
            
            results = []
            body_transl = bm.transl.clone().detach()
            
            # Step 2: Optimize the full model
            final_loss_val = 0
            for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
                opt_start = time.time()
                new_params = defaultdict(transl=body_transl, global_orient=orient, body_pose=body_mean_pose)
                bm.reset_params(**new_params)
                with torch.no_grad():
                    pose_embeddings.fill_(0)

                for opt_idx, curr_weights in enumerate(tqdm(self.opt_weights, desc='Stage')):
                    bm.transl.requires_grad = opt_idx in self.trans_opt_stages
                    body_params = list(bm.parameters())

                    final_params = list(filter(lambda x: x.requires_grad, body_params))
                    final_params.append(pose_embeddings)

                    body_optimizer, body_create_graph = optim_factory.create_optimizer(final_params, **args)
                    body_optimizer.zero_grad()

                    curr_weights['bending_prior_weight'] = (3.17 * curr_weights['body_pose_weight'])
                    if self.use_hands:
                        joint_weights[:, 25:76] = curr_weights['hand_weight']
                    if self.use_face:
                        joint_weights[:, 76:] = curr_weights['face_weight']
                    loss.reset_loss_weights(curr_weights)
                    closure = monitor.create_fitting_closure(
                        body_optimizer, bm,
                        camera=camera, gt_joints=gt_joints, ground_y=ground_y, cam_R=cam_R,
                        joints_conf=joints_conf, joint_weights=joint_weights,
                        loss=loss, create_graph=body_create_graph,
                        use_vposer=True, vposer=self.vposer,
                        pose_embedding=pose_embeddings,
                        scene_v=scene_v, scene_vn=scene_vn, scene_f=scene_f, 
                        ftov=ftov, return_verts=True, return_full_pose=True)

                    if self.interactive:
                        torch.cuda.synchronize()
                        stage_start = time.time()

                    final_loss_val = monitor.run_fitting(
                        body_optimizer, closure, final_params, bm,
                        pose_embedding=pose_embeddings, vposer=self.vposer, use_vposer=True)
               
                    if self.interactive:
                        torch.cuda.synchronize()
                        elapsed = time.time() - stage_start
                        tqdm.write('Stage {:03d} done after {:.4f} seconds, Loss:{}'.format(
                                opt_idx, elapsed, final_loss_val))

                if self.interactive:
                    torch.cuda.synchronize()
                    elapsed = time.time() - opt_start
                    tqdm.write('Body fitting Orientation {} done after {:.4f} seconds'.format(or_idx, elapsed))
                    tqdm.write('Body final loss val = {:.5f}'.format(final_loss_val))
                result = {'camera_' + str(key): val.detach().cpu().numpy()
                          for key, val in camera.named_parameters()}
                result.update({key: val.detach().cpu().numpy()
                               for key, val in bm.named_parameters()})
                result['pose_embedding'] = pose_embeddings.detach().cpu().numpy()
                body_pose = self.vposer.forward(pose_embeddings).view(-1, 63)
                result['body_pose'] = body_pose.detach().cpu().numpy()      
                result['keypoints'] = keypoints
                results.append({'loss': final_loss_val, 'result': result})
                
            if len(results) > 1:
                min_idx = (0 if results[0]['loss'] < results[1]['loss'] else 1)
            else:
                min_idx = 0
            result = results[min_idx]['result']
            for i in range(len(keypoints)):
                for key in result.keys():
                    self.results[i][key] = result[key][[i]]
        return final_loss_val
                    
    def get_body_vertices(self, convert=True):
        num_persons = len(self.results)
        vertices_all = []
        bm = self.create_single_body(create_body_pose=True)
        for i in range(num_persons):
            param_names = [key for key, _ in bm.named_parameters()]            
            torch_params = {
                name: to_tensor(self.results[i][name].reshape(1, -1), device=self.device)
                for name in param_names}

            model_output = bm(return_verts=True, **torch_params)
            vertices = model_output.vertices.detach().cpu().numpy().squeeze()
            if convert:
                # Convert to coordinate system used in Total3D
                vertices = vertices[:, [2, 1, 0]]
                vertices[:, 1] = - vertices[:, 1]
                vertices[:, 2] = - vertices[:, 2]
            vertices_all.append(vertices)
        return vertices_all


def to_tensor(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    else:
        return torch.from_numpy(value).to(device)