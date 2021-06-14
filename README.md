## Holistic 3D Human and Scene Mesh Estimation from Single View Images
[Paper](https://arxiv.org/pdf/2012.01591.pdf) | [Project Page](https://zzweng.github.io/holistic_mesh/)


## Sample Usage 
```
cd code
python main.py --config smpl_configs/config.yaml --recording_folder PATH_TO_DATA --resume_from_smplify True --out_folder PATH_TO_OUTPUT --data_path PATH_TO_DATA --scene_iters 10 --joint_iters 10
```
`PATH_TO_DATA` should contain folder `images` with RGB input files, folder `keypoints` with OpenPose output json files, and `detections.json` with 2D detections of the objects.


## Citation
If you find our work useful in your research, please cite our paper:
```
@article{weng2020holistic,
  title={Holistic 3D Human and Scene Mesh Estimation from Single View Images},
  author={Weng, Zhenzhen and Yeung, Serena},
  journal={arXiv preprint arXiv:2012.01591},
  year={2020}
}
```

## References
This repo builds upon previous great works ([PROX](https://github.com/mohamedhassanmus/prox) and [3DTotalUnderstanding](https://github.com/yinyunie/Total3DUnderstanding)), and borrowed scripts from their repo.
