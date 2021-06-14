import json, os, pickle
from collections import namedtuple

import cv2
import numpy as np
import torch
from torch.utils import data

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = ({}, None, None)


class Dataset(data.Dataset):
    """ Holds images, 2D keypoints.
    """ 
    def __init__(self, img_folder, keypoints_folder, fitting_dir=None): 
        """
        img_folder: path to images
        keypoints_folder: path to json files output by openpose
        """
        self.img_folder = img_folder
        self.kp_folder = keypoints_folder
        self.img_list = sorted(os.listdir(self.img_folder))
        self.fitting_dir = fitting_dir
 
        # count number of persons in each image
        num_persons = 0
        person_to_img_map = {}
        for img_name in self.img_list:
            kp_name = os.path.join(self.kp_folder, os.path.splitext(img_name)[0]+'_keypoints.json')
            kp = read_keypoints(kp_name)
            for p_i in range(len(kp)):
                # maps dataset entry id to the img name and the person idx in this img.
                person_to_img_map[num_persons+p_i] = (img_name, p_i)
            num_persons += len(kp)
        self.num_persons = num_persons
        self.person_to_img_map = person_to_img_map
        print('{} persons detected in {} images.'.format(num_persons, len(self.img_list)))

    def __len__(self):
        return self.num_persons

    def __getitem__(self, idx):
        img_name, p_i = self.person_to_img_map[idx]
        img_path = os.path.join(self.img_folder, img_name)
        img = cv2.imread(img_path)
        frame_name = os.path.splitext(img_name)[0]
        
        kp_name = os.path.join(self.kp_folder, frame_name+'_keypoints.json')
        kp = read_keypoints(kp_name)[p_i]
    
        if self.fitting_dir is not None:
            fitting_path = os.path.join(self.fitting_dir, frame_name, 'result.pkl')
            fitting = pickle.load(open(fitting_path, 'rb'))
        else:
            fitting = []
        return {'img_path': img_path, 'img': img, 'keypoints': kp, 'fitting': fitting}



def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False, dtype=torch.float32):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []
    gender_pd = []
    gender_gt = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        use_hands, use_face = True, True
        if use_hands:
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

        keypoints.append(torch.tensor(body_keypoints, dtype=dtype))

    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt).keypoints
