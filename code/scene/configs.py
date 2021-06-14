import os, pickle
import numpy as np

curr_path = os.path.dirname(os.path.abspath(__file__))
__layout_avg_file = os.path.join(curr_path, 'data', 'layout_avg_file.pkl')
__size_avg_path = os.path.join(curr_path, 'data', 'size_avg_category.pkl')


def get_bins():
    bin = {}
    # there are faithful priors for layout locations, we can use it for regression.
    if os.path.exists(__layout_avg_file):
        with open(__layout_avg_file, 'rb') as file:
            layout_avg_dict = pickle.load(file)
    else:
        raise IOError('No layout average file in %s. Please check.' % (__layout_avg_file))

    bin['layout_centroid_avg'] = layout_avg_dict['layout_centroid_avg']
    bin['layout_coeffs_avg'] = layout_avg_dict['layout_coeffs_avg']

    '''layout orientation bin'''
    NUM_LAYOUT_ORI_BIN = 2
    ORI_LAYOUT_BIN_WIDTH = np.pi / 4
    bin['layout_ori_bin'] = [[np.pi / 4 + i * ORI_LAYOUT_BIN_WIDTH, np.pi / 4 + (i + 1) * ORI_LAYOUT_BIN_WIDTH]
                             for i in range(NUM_LAYOUT_ORI_BIN)]

    '''camera bin'''
    PITCH_NUMBER_BINS = 2
    PITCH_WIDTH = 40 * np.pi / 180
    ROLL_NUMBER_BINS = 2
    ROLL_WIDTH = 20 * np.pi / 180

    # pitch_bin = [[-60 * np.pi/180, -20 * np.pi/180], [-20 * np.pi/180, 20 * np.pi/180]]
    bin['pitch_bin'] = [[-60.0 * np.pi / 180 + i * PITCH_WIDTH, -60.0 * np.pi / 180 + (i + 1) * PITCH_WIDTH] for
                        i in range(PITCH_NUMBER_BINS)]
    # roll_bin = [[-20 * np.pi/180, 0 * np.pi/180], [0 * np.pi/180, 20 * np.pi/180]]
    bin['roll_bin'] = [[-20.0 * np.pi / 180 + i * ROLL_WIDTH, -20.0 * np.pi / 180 + (i + 1) * ROLL_WIDTH] for i in
                       range(ROLL_NUMBER_BINS)]

    '''bbox orin, size and centroid bin'''
    # orientation bin
    NUM_ORI_BIN = 6
    ORI_BIN_WIDTH = float(2 * np.pi / NUM_ORI_BIN) # 60 degrees width for each bin.
    # orientation bin ranges from -np.pi to np.pi.
    bin['ori_bin'] = [[(i - NUM_ORI_BIN / 2) * ORI_BIN_WIDTH, (i - NUM_ORI_BIN / 2 + 1) * ORI_BIN_WIDTH] for i
                      in range(NUM_ORI_BIN)]

    if os.path.exists(__size_avg_path):
        with open(__size_avg_path, 'rb') as file:
            avg_size = pickle.load(file)
    else:
        raise IOError('No object average size file in %s. Please check.' % (__size_avg_path))

    bin['avg_size'] = np.vstack([avg_size[key] for key in range(len(avg_size))])

    # for each object bbox, the distance between camera and object centroid will be estimated.
    NUM_DEPTH_BIN = 6
    DEPTH_WIDTH = 1.0
    # centroid_bin = [0, 6]
    bin['centroid_bin'] = [[i * DEPTH_WIDTH, (i + 1) * DEPTH_WIDTH] for i in
                           range(NUM_DEPTH_BIN)]
    return bin
