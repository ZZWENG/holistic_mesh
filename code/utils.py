from itertools import product
import os
from time import time

from mesh_to_sdf import mesh_to_sdf
import cv2, pyrender, trimesh
import numpy as np
import torch
import PIL.Image as pil_img


def merge_verts_and_faces(nverts, nfaces, face_offset=0):
    N = len(nverts)
    assert len(nfaces) == N
    verts = np.concatenate(nverts, axis=0)

    faces = []
    off_so_far = 0
    for i in range(N):
        faces.append(nfaces[i] + off_so_far - face_offset)
        off_so_far += nverts[i].shape[0]
 
    faces = np.concatenate(faces, axis=0)
    return verts, faces


def generate_layout_points(points):
    points = np.concatenate([points, np.linspace(points[4], points[7], num=5)])
    points = np.concatenate([points, np.linspace(points[5], points[6], num=5)])
    points = np.concatenate([points, np.linspace(points[2], points[6], num=5)])
    points = np.concatenate([points, np.linspace(points[3], points[7], num=5)])
    samples = 1000
    points = np.array([np.random.normal(points[i], 0.05, size=(samples,3))
                       for i in range(points.shape[0])]).reshape(-1, 3)
    colors = np.array([[66., 66., 255.]] * (2*samples) + 
                  [[215, 66, 245]] * (2*samples) +
                  [[255., 66., 66.]] * (2*samples) +
                  [[104, 255, 66]] * (2*samples) +
                  [[255, 245, 61]] * (20*samples)
                 )/255.
    return points, colors


def render_image(img_path, out_image_path, out_models_path, body_verts, body_faces, 
                scene_verts=None, scene_faces=None, layout=None):
    img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
    img = cv2.flip(img, 1)
    
    if scene_verts is not None:
        scene_mesh = trimesh.Trimesh(scene_verts, scene_faces, process=False)
        scene_mesh.export(os.path.join(out_models_path, 'scene.ply'))
    else:
        scene_mesh = None

    if layout is not None:
        points, colors = generate_layout_points(layout)
        layout_points = pyrender.Mesh.from_points(points, colors=colors)
    else:
        layout_points = None

    body_verts, body_faces = merge_verts_and_faces(body_verts, body_faces)
    human_mesh = trimesh.Trimesh(body_verts, body_faces, process=False)
    human_mesh.export(os.path.join(out_models_path, 'body.ply'))
    
    camera_pose = np.eye(4)
    camera_center = np.array([951.30, 536.77])
    camera_pose = np.array([1.0, -1.0, 1.0, 1.0]).reshape(-1, 1) * camera_pose
    camera = pyrender.camera.IntrinsicsCamera(
        fx=1060.53, fy=1060.38, cx=camera_center[0], cy=camera_center[1])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    py_mesh = pyrender.Mesh.from_trimesh(human_mesh, material=material)
    ## rendering mesh
    H, W, _ = img.shape
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    scene.add(py_mesh, 'mesh')
    if scene_mesh is not None:
        py_mesh_scene = pyrender.Mesh.from_trimesh(scene_mesh, material=material)
        scene.add(py_mesh_scene, 'mesh_scene')
    if layout_points is not None:
        scene.add(layout_points, 'layout')

    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    valid_mask = (depth > 0)[:, :, np.newaxis]
    input_img = img
    output_img = (color * valid_mask + (1 - valid_mask) * input_img)
    pimg = pil_img.fromarray((output_img * 255).astype(np.uint8))
    pimg.save(os.path.join(out_image_path, 'rendered_{}'.format(img_path.split('/')[-1])))
    return output_img

def collect_results_for_image(b_model, batch):
    body_verts = b_model.get_body_vertices(convert=False)
    img_names = set(batch['img_path'])
    faces = b_model.bm.faces
    results = dict()
    for img_name in img_names:
        idx = np.where(np.array(batch['img_path'])==img_name)[0]
        
        results[img_name] = [
            [body_verts[i] for i in idx],
            [faces for i in idx]]
        res_pickle = []  # model params to be pickled
        for i in idx:
            res_pickle.append({'result': b_model.results[i]})
        results[img_name].append(res_pickle)
    return results


def get_sdf(verts, faces, grid_dim, vmin, vmax):
    mesh = trimesh.Trimesh(verts, faces, process=False)
    d1 = torch.linspace(vmin[0], vmax[0], grid_dim)
    d2 = torch.linspace(vmin[1], vmax[1], grid_dim)
    d3 = torch.linspace(vmin[2], vmax[2], grid_dim)
    meshx, meshy, meshz = torch.meshgrid((d1, d2, d3))
    qp = {
        (i,j,h): (meshx[i,j,h].item(), meshy[i,j,h].item(), meshz[i,j,h].item()) 
        for (i,j,h) in product(range(grid_dim), range(grid_dim), range(grid_dim))
    }
    qp_idxs = list(qp.keys())
    qp_values = np.array(list(qp.values()))

    t = time()
    qp_sdfs = mesh_to_sdf(mesh, qp_values)  # 10 secs
    qp_map = {qp_idxs[k]: qp_sdfs[k] for k in range(len(qp_sdfs))}
    qp_sdfs = np.zeros((grid_dim, grid_dim, grid_dim))

    for (i,j,h) in product(range(grid_dim), range(grid_dim), range(grid_dim)):
        qp_sdfs[i,j,h] = qp_map[(i,j,h)]

    qp_sdfs = torch.tensor(qp_sdfs)
#     print('Generated grid sdf in {} secs'.format(time()-t))
    return qp_sdfs
