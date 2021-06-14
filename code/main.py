import argparse, json, sys, os, pickle

import cv2, torch, trimesh
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from scipy.io import savemat

from cmd_parser import parse_config
from dataset import Dataset
from utils import render_image, collect_results_for_image, get_sdf
from body.body_model import BodyModel
from scene_model import SceneModel


def main(args):
    base_folder = args['recording_folder']
    out_folder = args['out_folder']
    data_path = args['data_path']
    fitting_dir = args["resume_from_dir"] if args['resume_from_smplify'] else None
    batch_size = args['batch_size']
    
    dataset = Dataset(img_folder=os.path.join(base_folder, 'images'), 
        keypoints_folder=os.path.join(base_folder, 'keypoints'),
        fitting_dir=fitting_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    out_models_path = os.path.join(out_folder, 'models')
    out_image_path = os.path.join(out_folder, 'images')
    if not os.path.exists(out_models_path):
        os.makedirs(out_models_path)
    if not os.path.exists(out_image_path):
        os.makedirs(out_image_path)

    b_model = BodyModel(args)
    s_model = SceneModel(data_path, b_model.device)
    s_model.load_weight(r'scene/pretrained_model.pth')
    writer = SummaryWriter(os.path.join(out_folder, 'logs'))
    
    for batch_i, batch in enumerate(dataloader):
        if not args['resume_from_smplify']:
            # fit body models
            b_model.fit(batch, args['maxiters'])
        else:
            b_model.results = [batch["fitting"][0]['result']]
        
        # optimize scene with within-scene losses
        loss_dicts = s_model.optimize(is_joint=False, iters=args['scene_iters'], 
                                      image_path=batch['img_path'][0], 
                                      sdf_pen_weight=0.1, object_groud_weight=10)
        for i, loss_dict in enumerate(loss_dicts):
            for key, loss_val in loss_dict.items():
                writer.add_scalar('scene:'+key, loss_val, 
                                  batch_size*batch_i*args['scene_iters']+i)
        
        # optimize with body-scene losses
        opt_joint = args['joint_iters'] > 0
        if opt_joint:
            # fit scene with human information
            body_verts = b_model.get_body_vertices(convert=True)
            body_faces = b_model.bm.faces
            loss_dicts = s_model.optimize(
                is_joint=True, iters=args['joint_iters'], image_path=batch['img_path'][0],
                body_verts=body_verts, body_faces=body_faces,
                sdf_pen_weight=args['sdf_scene_weight'], object_groud_weight=args['ground_plane_weight'], 
                body_ground_weight=args['ground_plane_weight'])
            
            # log losses
            for i, loss_dict in enumerate(loss_dicts):
                for key, loss_val in loss_dict.items():
                    writer.add_scalar('joint:'+key, loss_val, 
                                      batch_size*batch_i*args['joint_iters']+i)
            
            # fit human with scene information
            verts, faces, cam_R, lo_bdb3D_out, lo_bdb3D_cam_coord \
                = s_model.inference(image_path=batch['img_path'][0])
            ground_y = lo_bdb3D_out[:, 1].min()
            vmin = torch.from_numpy(verts.min(0)).to(b_model.device)
            vmax = torch.from_numpy(verts.max(0)).to(b_model.device)
            grid_dim = args['sdf_grid_dim']
            qp_sdfs = get_sdf(verts, faces, grid_dim, vmin, vmax).to(b_model.device).type(torch.float32)
            b_model.fit(batch, args['maxiters'], verts, faces, qp_sdfs, vmin, vmax, grid_dim, cam_R, ground_y)
        
        # render images and save results
        results_by_img_name = collect_results_for_image(b_model, batch)
        for img_path, res in results_by_img_name.items():
            img_arr = render_image(
               img_path, out_image_path, out_models_path,
               res[0], res[1] #, verts, faces, lo_bdb3D_cam_coord
            )
            res_out_path = os.path.join(out_models_path, os.path.splitext(img_path.split('/')[-1])[0])
            print('Saved result.pkl to ', res_out_path)
            os.makedirs(res_out_path, exist_ok=True)
            with open(os.path.join(res_out_path, "result.pkl"), 'wb') as result_file:
                pickle.dump(res[2], result_file, protocol=2)

        writer.add_image('images', torch.from_numpy(img_arr).permute(2,0,1), batch_i)
                
    writer.close()

    
if __name__ == '__main__':
    args = parse_config()
    main(args)


