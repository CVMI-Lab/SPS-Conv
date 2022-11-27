import torch
import torch.nn as nn
import spconv.pytorch as spconv
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
import cv2
import numpy as np

def separate_foreground(x, batch_dict, voxel_stride, voxel_size, point_cloud_range):
    indices = x.indices
    features = x.features
    point_cloud_range = torch.Tensor(point_cloud_range).cuda()
    voxel_size = torch.Tensor(voxel_size).cuda()
    inv_idx =  torch.Tensor([2, 1, 0]).long().cuda()
    spatial_indices = x.indices[:, 1:] * voxel_stride
    voxels_3d = spatial_indices * voxel_size + point_cloud_range[:3]
    batch_size = x.batch_size
    mask_voxels = []
    box_of_pts_cls_targets = []
    for b in range(batch_size):
        index=x.indices[:, 0]
        batch_index = index == b
        voxels_3d_batch = voxels_3d[batch_index].unsqueeze(0)
        gt_boxes = batch_dict['gt_boxes'][b, :, :7].unsqueeze(0)
        box_of_pts_batch = points_in_boxes_gpu(voxels_3d_batch[:, :, inv_idx], gt_boxes).squeeze(0)
        box_of_pts_cls_targets.append(box_of_pts_batch>=0)
    
    
    box_of_pts_cls_targets = torch.cat(box_of_pts_cls_targets)
    new_x = spconv.SparseConvTensor(features[box_of_pts_cls_targets], indices[box_of_pts_cls_targets], x.spatial_shape, x.batch_size)
    # print("new features shape:", features[box_of_pts_cls_targets].shape, "features shape:", features.shape)
    # visualize(batch_dict, index[box_of_pts_cls_targets], voxels_3d[box_of_pts_cls_targets], "fore")
    # visualize(batch_dict, index, voxels_3d, "all")
    # visualize(batch_index, voxels_3d)
    return new_x

def write_obj(points, colors, out_filename):
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i]
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()

def visualize(batch_dict, index, voxels_3d, base_name):
    # calibs = batch_dict['calib']
    batch_size = batch_dict['batch_size']
    # h, w = batch_dict['images'].shape[2:]
    # batch_size = batch_dict['batch_size']
    inv_idx = [2, 1, 0]
    for b in range(batch_size):
        # calib = calibs[b]
        voxels_3d_batch = voxels_3d[index==b]
        # voxels_2d, _ = calib.lidar_to_img(voxels_3d[:, inv_idx].cpu().numpy())

        _color = cv2.COLORMAP_PARULA
        grad = np.uint8(np.ones((voxels_3d_batch.shape[0], 3)))
        grad = cv2.applyColorMap(np.uint8(50 * grad), _color)[:, 0, :] #cv2.COLORMAP_HOT)
        # image = batch_dict['images'][b]
        # image = image.cpu().permute(1, 2, 0).cpu().numpy()
        # image = (image * 255).astype(np.uint8).copy()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        base_path = '/data/home/jianhuiliu/newData/Research/CVMI-3D-Pruning/3d-detect-pruning-develop/output/test_visual/'
        write_obj(voxels_3d_batch, grad, base_path + base_name +'_%s_point.obj'%(batch_dict['frame_id'][b]))

        # test_img = image.copy()
        # for i, _point in enumerate(voxels_2d):
        #     cv2.circle(test_img, tuple(_point.astype(np.int)), 2, tuple(grad[i].tolist()), -1)

        # base_path = '/data/home/jianhuiliu/newData/Research/CVMI-3D-Pruning/3d-detect-pruning-develop/output/test_visual/'
        # cv2.imwrite(base_path + '%s_down_img_%s_point.png'%(indice_key, batch_dict['frame_id'][b]), test_img)


def sort_by_indices(features_foreground_cat, indices_foreground_coords, additional_features=None):
    a = indices_foreground_coords[:, 1:]
    # print("a shape:", a.shape)
    augmented_a = a.select(1, 0) * a[:, 1].max() * a[:, 2].max() + a.select(1, 1) * a[:, 2].max() + a.select(1, 2)
    augmented_a_sorted, ind = augmented_a.sort()
    features_foreground_cat = features_foreground_cat[ind]
    indices_foreground_coords = indices_foreground_coords[ind]
    if not additional_features is None:
        additional_features = additional_features[ind]
    return features_foreground_cat, indices_foreground_coords, additional_features

def check_repeat(x_foreground_features, x_foreground_indices, additional_features=None, sort_first=True, flip_first=True):
    if sort_first:
        x_foreground_features, x_foreground_indices, additional_features = sort_by_indices(x_foreground_features, x_foreground_indices, additional_features)

    if flip_first:
        x_foreground_features, x_foreground_indices = x_foreground_features.flip([0]), x_foreground_indices.flip([0])

    if not additional_features is None:
        additional_features=additional_features.flip([0])

    a = x_foreground_indices[:, 1:].int()
    augmented_a = torch.add(torch.add(a.select(1, 0) * a[:, 1].max() * a[:, 2].max(), a.select(1, 1) * a[:, 2].max()), a.select(1, 2))
    _unique, inverse, counts = torch.unique_consecutive(augmented_a, return_inverse=True, return_counts=True, dim=0)
    
    if _unique.shape[0] < x_foreground_indices.shape[0]:
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        x_foreground_features_new = torch.zeros((_unique.shape[0], x_foreground_features.shape[-1]), device=x_foreground_features.device)
        x_foreground_features_new.index_add_(0, inverse.long(), x_foreground_features)
        x_foreground_features = x_foreground_features_new
        perm_ = inverse.new_empty(_unique.size(0)).scatter_(0, inverse, perm)
        x_foreground_indices = x_foreground_indices[perm_].int()

        if not additional_features is None:
            additional_features_new = torch.zeros((_unique.shape[0],), device=additional_features.device)
            additional_features_new.index_add(0, inverse.long(), additional_features)
            additional_features = additional_features_new / counts
    return x_foreground_features, x_foreground_indices, additional_features


def split_voxels(x, b, offsets_3d, voxels_3d, kernel_offsets, mask_multi=True, topk=True, threshold=0.5):
    index = x.indices[:, 0]
    batch_index = index==b
    indices_ori = x.indices[batch_index]
    features_ori = x.features[batch_index]
    mask_voxel = offsets_3d[batch_index, -1].sigmoid()
    mask_voxel_kernel = offsets_3d[batch_index, :-1].sigmoid()

    # resampling according to the mask

    if mask_multi:
        features_ori *= mask_voxel.unsqueeze(-1)

    if topk:
        _, indices = mask_voxel.sort(descending=True)
        indices_foreground = indices[:int(mask_voxel.shape[0]*threshold)]
        indices_background = indices[int(mask_voxel.shape[0]*threshold):]
    else:
        indices_foreground = mask_voxel > threshold
        indices_background = mask_voxel <= threshold

    features_foreground = features_ori[indices_foreground]
    indices_foreground_coords = indices_ori[indices_foreground]

    mask_voxel_kernel_foreground = mask_voxel_kernel[indices_foreground]
    mask_voxel_kernel_bool = mask_voxel_kernel_foreground>=threshold
    voxel_kerels_offset = kernel_offsets.unsqueeze(0).repeat(mask_voxel_kernel_bool.shape[0],1, 1)
    mask_voxel_kernel_selected = mask_voxel_kernel[indices_foreground][mask_voxel_kernel_bool]
    indices_foreground_kernels = indices_foreground_coords[:, 1:].unsqueeze(1).repeat(1, 26, 1)
    indices_with_offset = indices_foreground_kernels + voxel_kerels_offset
    selected_indices_with_offset = indices_with_offset[mask_voxel_kernel_bool]
    spatial_indices = (selected_indices_with_offset[:, 0] >0) * (selected_indices_with_offset[:, 1] >0) * (selected_indices_with_offset[:, 2] >0)  * \
                        (selected_indices_with_offset[:, 0] < x.spatial_shape[0]) * (selected_indices_with_offset[:, 1] < x.spatial_shape[1]) * (selected_indices_with_offset[:, 2] < x.spatial_shape[2])
    selected_indices_with_offset = selected_indices_with_offset[spatial_indices]
    selected_indices_with_offset = torch.cat([torch.ones((selected_indices_with_offset.shape[0], 1), device=features_foreground.device)*b, selected_indices_with_offset], dim=1)

    selected_features = torch.zeros((selected_indices_with_offset.shape[0], features_ori.shape[1]), device=features_foreground.device)

    ##
    #selected_features, selected_indices_with_offset, mask_voxel_kernel_selected = check_repeat2(selected_features, selected_indices_with_offset, additional_features=mask_voxel_kernel_selected)

    features_foreground_cat = torch.cat([features_foreground, selected_features], dim=0)
    indices_foreground_coords = torch.cat([indices_foreground_coords, selected_indices_with_offset], dim=0)
    mask_voxel_kernel_selected = torch.cat([torch.ones(features_foreground.shape[0], device=features_foreground.device), mask_voxel_kernel_selected], dim=0)

    # sort_by_indices()
    features_foreground_out, indices_foreground_coords_out, mask_voxel_kernel_selected_out = check_repeat(features_foreground_cat, indices_foreground_coords, additional_features=mask_voxel_kernel_selected)
    #raise ValueError('Stop.')

    features_background = features_ori[indices_background]
    indices_background = indices_ori[indices_background]

    return features_foreground_out, indices_foreground_coords_out, features_background, indices_background, mask_voxel_kernel_selected_out

def split_voxels_v2(x, b, voxel_importance, kernel_offsets, mask_multi=True, pruning_mode="topk", pruning_ratio=0.5):
    index = x.indices[:, 0]
    batch_index = index==b
    indices_ori = x.indices[batch_index]
    features_ori = x.features[batch_index]
    voxel_importance = voxel_importance[batch_index]

    if mask_multi:
        features_ori *= voxel_importance

    # get mask
    # print("pruning_mode-----------------------:", pruning_mode)
    if pruning_mode == "topk":
        _, indices = voxel_importance.view(-1,).sort()
        indices_im = indices[int(voxel_importance.shape[0]*pruning_ratio):]
        indices_nim = indices[:int(voxel_importance.shape[0]*pruning_ratio)]
        # print("indices_im num:", indices_im.sum(), "indices_nim num:",indices_nim.sum(), "pruning_ratio:", pruning_ratio, "x shape:", x.features.shape)
        # print("indices_im num:", indices_im.shape, "indices_nim num:",indices_nim.shape, "pruning_ratio:", pruning_ratio, "x shape:", x.features.shape)
    elif pruning_mode == "thre":
        indices_im = (voxel_importance.view(-1,) > pruning_ratio)
        indices_nim = (voxel_importance.view(-1,) <= pruning_ratio)
        # print("indices_im num:", indices_im.sum(), "indices_nim num:",indices_nim.sum(), "pruning_ratio:", pruning_ratio, "x shape:", x.features.shape)
    
    features_im = features_ori[indices_im]
    coords_im = indices_ori[indices_im]
    voxel_kerels_offset = kernel_offsets.unsqueeze(0).repeat(features_im.shape[0],1, 1) # [features_im.shape[0], 26, 3]
    indices_im_kernels = coords_im[:, 1:].unsqueeze(1).repeat(1, kernel_offsets.shape[0], 1) # [coords_im.shape[0], 26, 3]
    # print("kernel_offsets:", kernel_offsets.dtype, "indices_im_kernels:", indices_im_kernels.dtype, "voxel_kerels_offset:", voxel_kerels_offset.dtype)
    indices_with_imp = (indices_im_kernels + voxel_kerels_offset).view(-1, 3)
    spatial_indices = (indices_with_imp[:, 0] >0) * (indices_with_imp[:, 1] >0) * (indices_with_imp[:, 2] >0)  * \
                        (indices_with_imp[:, 0] < x.spatial_shape[0]) * (indices_with_imp[:, 1] < x.spatial_shape[1]) * (indices_with_imp[:, 2] < x.spatial_shape[2])
    
    selected_indices = indices_with_imp[spatial_indices]
    selected_indices = torch.cat([torch.ones((selected_indices.shape[0], 1), device=features_im.device, dtype=torch.int)*b, selected_indices], dim=1)
    selected_features = torch.zeros((selected_indices.shape[0], features_ori.shape[1]), device=features_im.device)
    
    features_im = torch.cat([features_im, selected_features], dim=0) # [N', C]
    coords_im = torch.cat([coords_im, selected_indices], dim=0) # [N', 3]
    # mask_kernel_im = voxel_importance[indices_im][spatial_indices]
    # mask_kernel_im = mask_kernel_im.unsqueeze(1).repeat(1, kernel_offsets.shape[0], 1) 
    # mask_kernel_im = torch.cat([torch.ones(features_im_cat.shape[0], device=features_im.device), mask_kernel_im], dim=0)
    # print("before:", features_im.shape)
    assert features_im.shape[0] == coords_im.shape[0]
    if indices_im.sum()>0:
        features_im, coords_im, _ = check_repeat(features_im, coords_im)
        # print("after:", features_im.shape)
    # print("coords_im after:", coords_im.dtype)
    features_nim = features_ori[indices_nim]
    coords_nim = indices_ori[indices_nim]
    
    return features_im, coords_im, features_nim, coords_nim