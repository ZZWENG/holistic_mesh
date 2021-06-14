# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import configargparse


def parse_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter

    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'PyTorch implementation of SMPLifyX'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='SMPLifyX')
    ### Begin: Added parameters
    parser.add_argument('--recording_folder', type=str, default='',
                        help='The folder where the preprocessd recording files are stored')
    parser.add_argument('--body_segments_dir', type=str, default='/home/groups/syyeung/behavioral_coding/body_segments/',
                        help='The folder where the body segments files (json) are stored')
    parser.add_argument('--data_path', type=str, default='',
                        help='The folder where the preprocessd recording files are stored')
    parser.add_argument('--out_folder', type=str, default='',
                        help='The folder where the fitting results are stored.')
    parser.add_argument('--sdf_grid_dim', type=int, default=8,
                       help='Grid dim when calculating mesh sdf.')
    parser.add_argument('--resume_from_smplify', type=lambda arg: arg.lower() == 'true', default=False,
                        help='Whether to load the fitting results from smplify and continue from there.')
    parser.add_argument('--resume_from_dir', type=str, default='',
                        help='The folder where the intermediate fitting result files are stored.')
    parser.add_argument('--joint_iters', type=int, default=10,
                       help='Number of iterations to run joint optimization per image.')
    parser.add_argument('--scene_iters', type=int, default=5,
                       help='Number of iterations to run joint optimization per image.')
    parser.add_argument('--sdf_scene_weight', type=int, default=100)
    parser.add_argument('--ground_plane_weight', type=int, default=100)

    ### End

    parser.add_argument('-c', '--config',
                        required=True, is_config_file=True,
                        help='config file path')
    parser.add_argument('--loss_type', default='smplify', type=str,
                        help='The type of loss to use')
    parser.add_argument('--interactive',
                        type=lambda arg: arg.lower() == 'true',
                        default=False,
                        help='Print info messages during the process')
    parser.add_argument('--degrees', type=float, default=[0, 90, 180, 270],
                        help='Degrees of rotation for rendering the final result')
    parser.add_argument('--joints_to_ign', default=-1, type=int,
                        nargs='*', help='Indices of joints to be ignored')
    parser.add_argument('--gender_lbl_type', default='none',
                        choices=['none', 'gt', 'pd'], type=str,
                        help='The type of gender label to use')
    parser.add_argument('--gender', type=str, default='neutral',
                        choices=['neutral', 'male', 'female'],
                        help='Use gender neutral or gender specific SMPL model')
    parser.add_argument('--model_type', default='smpl', type=str, choices=['smpl', 'smplh', 'smplx'],
                        help='The type of the model that we will fit to the  data.')
    parser.add_argument('--camera_type', type=str, default='persp', choices=['persp'],
                        help='The type of camera used')
    parser.add_argument('--optim_jaw', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='Optimize over the jaw pose')
    parser.add_argument('--optim_hands', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Optimize over the hand pose')
    parser.add_argument('--optim_expression', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Optimize over the expression')
    parser.add_argument('--optim_shape', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Optimize over the shape space')

    parser.add_argument('--model_folder', default='models', type=str,
                        help='The directory where the models are stored.')
    parser.add_argument('--use_joints_conf', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='Use the confidence scores for the optimization')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='The size of the batch (how many persons to fit at a time)')
    parser.add_argument('--num_gaussians', default=8, type=int,
                        help='The number of gaussian for the Pose Mixture Prior.')
    parser.add_argument('--use_pca', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the low dimensional PCA space for the hands')
    parser.add_argument('--num_pca_comps', default=6, type=int,
                        help='The number of PCA components for the hand.')
    parser.add_argument('--flat_hand_mean', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use the flat hand as the mean pose')
    parser.add_argument('--body_prior_type', default='mog', type=str,
                        help='The type of prior that will be used to' +
                        ' regularize the optimization. Can be a Mixture of' +
                        ' Gaussians (mog)')
    parser.add_argument('--left_hand_prior_type', default='mog', type=str,
                        choices=['mog', 'l2', 'None'],
                        help='The type of prior that will be used to' +
                        ' regularize the optimization of the pose of the' +
                        ' left hand. Can be a Mixture of' +
                        ' Gaussians (mog)')
    parser.add_argument('--right_hand_prior_type', default='mog', type=str,
                        choices=['mog', 'l2', 'None'],
                        help='The type of prior that will be used to' +
                        ' regularize the optimization of the pose of the' +
                        ' right hand. Can be a Mixture of' +
                        ' Gaussians (mog)')
    parser.add_argument('--jaw_prior_type', default='l2', type=str,
                        choices=['l2', 'None'],
                        help='The type of prior that will be used to' +
                        ' regularize the optimization of the pose of the' +
                        ' jaw.')
    parser.add_argument('--use_vposer', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use the VAE pose embedding')
    parser.add_argument('--vposer_ckpt', type=str, default='',
                        help='The path to the V-Poser checkpoint')
    # Left/Right shoulder and hips
    parser.add_argument('--init_joints_idxs', nargs='*', type=int,
                        default=[9, 12, 2, 5],
                        help='Which joints to use for initializing the camera')
    parser.add_argument('--body_tri_idxs', default='5.12,2.9',
                        type=lambda x: [list(map(int, pair.split('.')))
                                        for pair in x.split(',')],
                        help='The indices of the joints used to estimate' +
                        ' the initial depth of the camera. The format' +
                        ' should be vIdx1.vIdx2,vIdx3.vIdx4')

    parser.add_argument('--prior_folder', type=str, default='prior',
                        help='The folder where the prior is stored')
    parser.add_argument('--rho', default=100, type=float,
                        help='Value of constant of robust loss')
    parser.add_argument('--interpenetration',
                        default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to use the interpenetration term')
    parser.add_argument('--penalize_outside',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Penalize outside')
    parser.add_argument('--data_weights', nargs='*',
                        default=[1, ] * 5, type=float,
                        help='The weight of the data term')
    parser.add_argument('--body_pose_prior_weights',
                        default=[4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78],
                        nargs='*',
                        type=float,
                        help='The weights of the body pose regularizer')
    parser.add_argument('--shape_weights',
                        default=[1e2, 5 * 1e1, 1e1, .5 * 1e1],
                        type=float, nargs='*',
                        help='The weights of the Shape regularizer')
    parser.add_argument('--expr_weights',
                        default=[1e2, 5 * 1e1, 1e1, .5 * 1e1],
                        type=float, nargs='*',
                        help='The weights of the Expressions regularizer')
    parser.add_argument('--face_joints_weights',
                        default=[0.0, 0.0, 0.0, 2.0], type=float,
                        nargs='*',
                        help='The weights for the facial keypoints' +
                        ' for each stage of the optimization')
    parser.add_argument('--hand_joints_weights',
                        default=[0.0, 0.0, 0.0, 2.0],
                        type=float, nargs='*',
                        help='The weights for the 2D joint error of the hands')
    parser.add_argument('--jaw_pose_prior_weights',
                        nargs='*',
                        help='The weights of the pose regularizer of the' +
                        ' hands')
    parser.add_argument('--hand_pose_prior_weights',
                        default=[1e2, 5 * 1e1, 1e1, .5 * 1e1],
                        type=float, nargs='*',
                        help='The weights of the pose regularizer of the' +
                        ' hands')
    parser.add_argument('--coll_loss_weights',
                        default=[0.0, 0.0, 0.0, 2.0], type=float,
                        nargs='*',
                        help='The weight for the collision term')

    parser.add_argument('--depth_loss_weight', default=1e2, type=float,
                        help='The weight for the regularizer for the' +
                        ' z coordinate of the camera translation')
    parser.add_argument('--df_cone_height', default=0.5, type=float,
                        help='The default value for the height of the cone' +
                        ' that is used to calculate the penetration distance' +
                        ' field')
    parser.add_argument('--max_collisions', default=8, type=int,
                        help='The maximum number of bounding box collisions')
    parser.add_argument('--point2plane', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use point to plane distance')
    parser.add_argument('--part_segm_fn', default='', type=str,
                        help='The file with the part segmentation for the' +
                        ' faces of the model')
    parser.add_argument('--ign_part_pairs', default=None,
                        nargs='*', type=str,
                        help='Pairs of parts whose collisions will be ignored')
    parser.add_argument('--use_hands', default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the hand keypoints in the SMPL' +
                        'optimization process')
    parser.add_argument('--use_face', default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the facial keypoints in the optimization' +
                        ' process')
    parser.add_argument('--use_face_contour', default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the dynamic contours of the face')
    parser.add_argument('--side_view_thsh',
                        default=25,
                        type=float,
                        help='This is thresholding value that determines' +
                        ' whether the human is captured in a side view.' +
                        'If the pixel distance between the shoulders is less' +
                        ' than this value, two initializations of SMPL fits' +
                        ' are tried.')
    parser.add_argument('--optim_type', type=str, default='adam',
                        help='The optimizer used')
    parser.add_argument('--lr', type=float, default=1e-6,
                        help='The learning rate for the algorithm')
    parser.add_argument('--gtol', type=float, default=1e-8,
                        help='The tolerance threshold for the gradient')
    parser.add_argument('--ftol', type=float, default=2e-9,
                        help='The tolerance threshold for the function')
    parser.add_argument('--maxiters', type=int, default=100,
                        help='The maximum iterations for the optimization')
    #######################################################################
    ### PROX
    parser.add_argument('--frame_ids',
                        default=None, type=int,
                        nargs='*',
                        help='')
    parser.add_argument('--start', type=int, default=0,
                        help='id of the starting frame')
    parser.add_argument('--step', type=int, default=1,
                        help='step')

    parser.add_argument('--flip', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='flip image and keypoints')
    parser.add_argument('--camera_mode', type=str, default='moving',
                        choices=['moving', 'fixed'],
                        help='The mode of camera used')
    parser.add_argument('--focal_length_x',
                        default=5000,
                        type=float,
                        help='Value of focal length.')
    parser.add_argument('--focal_length_y',
                        default=5000,
                        type=float,
                        help='Value of focal length.')
    parser.add_argument('--camera_center_x',
                        default=None,
                        type=float,
                        help='Value of camera center x.')
    parser.add_argument('--camera_center_y',
                        default=None,
                        type=float,
                        help='Value of camera center y.')
    parser.add_argument('--trans_opt_stages',
                        default=[3,4], type=int,
                        nargs='*',
                        help='stages where translation will be optimized')
    ## Depth fitting
    parser.add_argument('--s2m_weights', default=[0.0, 0.0, 0.0, 0.0, 0.0], nargs='*', type=float,
                        help='')
    parser.add_argument('--s2m',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='Whether to save the meshes')
    parser.add_argument('--m2s',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='Whether to save the meshes')
    parser.add_argument('--m2s_weights', default=[0.0, 0.0, 0.0, 0.0, 0.0], nargs='*', type=float,
                        help='')
    parser.add_argument('--rho_s2m',
                        default=1,
                        type=float,
                        help='Value of constant of robust loss')
    parser.add_argument('--rho_m2s',
                        default=1,
                        type=float,
                        help='Value of constant of robust loss')
    parser.add_argument('--read_depth', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Read depth frames')
    parser.add_argument('--read_mask', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Read masks')
    parser.add_argument('--mask_folder', type=str, default='BodyIndex',
                        help='The folder where the keypoints are stored')
    parser.add_argument('--mask_on_color', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')
    parser.add_argument('--init_mode', default=None, type=str,
                        choices=[None, 'scan', 'both'],
                        help='')
    ################################
    # sdf penetration
    parser.add_argument('--sdf_penetration', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')
    parser.add_argument('--sdf_penetration_weights', default=[0.0, 0.0, 0.0, 0.0, 0.0], nargs='*', type=float,
                        help='')
    ## contact
    parser.add_argument('--contact',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--rho_contact',
                        default=1,
                        type=float,
                        help='Value of constant of robust loss')
    parser.add_argument('--contact_angle',
                        default=45,
                        type=float,
                        help='used to refine normals. (angle in degrees)')
    parser.add_argument('--contact_loss_weights',
                        default=[0.0, 0.0, 0.0, 0.0, 0.0], type=float,
                        nargs='*',
                        help='The weight for the contact term')
    parser.add_argument('--contact_body_parts',
                        default=['L_Leg', 'R_Leg', 'L_Hand', 'R_Hand', 'gluteus', 'back', 'thighs'], type=str,
                        nargs='*',
                        help='')
    parser.add_argument('--load_scene', type=lambda arg: arg.lower() in ['true', '1'],
                        default=False, help='')



    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict
