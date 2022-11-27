from functools import partial

import torch.nn as nn
import spconv.pytorch as spconv
from spconv.core import ConvAlgo

from ...utils.spconv_utils import replace_feature, spconv
from ...models.model_utils.pruning_block import SpatialPrunedSubmConvBlock, SpatialPrunedConvDownsample, SparseSequentialBatchdict


class PostActBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, voxel_stride=1, indice_key=None, stride=1, padding=0, pruning_ratio=0.5, point_cloud_range=[-3, -40, 0, 1, 40, 70.4], voxel_size= [0.1, 0.05, 0.05],
                   conv_type='subm', norm_fn=None, algo=ConvAlgo.Native, pruning_mode="topk", downsample_pruning_mode="thre"):
        super().__init__()
        self.indice_key = indice_key
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if conv_type == 'subm':
            self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        elif conv_type == 'subm_1':
            self.conv = spconv.SubMConv3d(in_channels, out_channels, 1, bias=False, indice_key=indice_key)
        elif conv_type == 'spconv':
            self.conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    bias=False, indice_key=indice_key)
        elif conv_type == 'inverseconv':
            self.conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        elif conv_type == "sprs":
            self.conv = SpatialPrunedConvDownsample(in_channels, out_channels, kernel_size, stride=stride, padding=padding, indice_key=indice_key, bias=False, 
                pruning_ratio=pruning_ratio, pred_mode="attn_pred", pred_kernel_size=None,  algo=algo, pruning_mode=downsample_pruning_mode,
                point_cloud_range=point_cloud_range, voxel_size=voxel_size, voxel_stride=voxel_stride)
        elif conv_type == "spss":
            self.conv = SpatialPrunedSubmConvBlock(
                in_channels, out_channels, kernel_size,  voxel_stride, stride=stride, padding=padding, indice_key=indice_key, bias=False, 
                pruning_ratio=pruning_ratio, pred_mode="attn_pred", pred_kernel_size=None, point_cloud_range=point_cloud_range, voxel_size=voxel_size, algo=algo, pruning_mode=pruning_mode)
        else:
            raise NotImplementedError        

        self.bn1 = norm_fn(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, batch_dict):
        if isinstance(self.conv, (SpatialPrunedSubmConvBlock,)) or isinstance(self.conv, (SpatialPrunedConvDownsample,)):
            x, batch_dict = self.conv(x, batch_dict)
        else:
            x = self.conv(x)
            
        x = replace_feature(x, self.bn1(x.features))
        x = replace_feature(x, self.relu(x.features))
        return x, batch_dict
        
        

def conv_block(in_channels, out_channels, kernel_size, voxel_stride=1, indice_key=None, stride=1, bias=False, padding=0, pruning_ratio=0.5, point_cloud_range=[-3, -40, 0, 1, 40, 70.4], voxel_size= [0.1, 0.05, 0.05],
                   conv_type='subm', norm_fn=None, algo=ConvAlgo.Native, pruning_mode="topk", downsample_pruning_mode="thre"):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=bias, indice_key=indice_key)
    elif conv_type == 'subm_1':
        conv = spconv.SubMConv3d(in_channels, out_channels, 1, bias=bias, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                bias=bias, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=bias)
    elif conv_type == "sprs":
        conv = SpatialPrunedConvDownsample(in_channels, out_channels, kernel_size, stride=stride, padding=padding, indice_key=indice_key, bias=False, 
            pruning_ratio=pruning_ratio, pred_mode="attn_pred", pred_kernel_size=None, algo=algo, pruning_mode=downsample_pruning_mode)
    elif conv_type == "spss":
        conv = SpatialPrunedSubmConvBlock(
            in_channels, out_channels, kernel_size,  voxel_stride, stride=stride, padding=padding, indice_key=indice_key, bias=bias, 
            pruning_ratio=pruning_ratio, pred_mode="attn_pred", pred_kernel_size=None, point_cloud_range=point_cloud_range, voxel_size=voxel_size, algo=algo, pruning_mode=pruning_mode)
    else:
        raise NotImplementedError     
    
    return conv


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, voxel_stride=1, indice_key=None, stride=1, padding=0, pruning_ratio=0.5, point_cloud_range=[-3, -40, 0, 1, 40, 70.4], voxel_size= [0.1, 0.05, 0.05],
                   conv_types=['subm', 'subm'], norm_fn=None, downsample=None, algo=ConvAlgo.Native, pruning_mode="topk"):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.indice_key = indice_key
        self.inplanes = inplanes
        self.planes = planes
        
        self.conv1 = conv_block(
            inplanes, planes, 3, voxel_stride=voxel_stride, norm_fn=norm_fn, padding=padding, bias=bias, indice_key=indice_key+"_1", conv_type= conv_types[0], pruning_ratio=pruning_ratio,
                  point_cloud_range=point_cloud_range, voxel_size=voxel_size, algo=algo, pruning_mode=pruning_mode
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()

        self.conv2 = conv_block(
            planes, planes, 3, voxel_stride=voxel_stride, norm_fn=norm_fn, padding=padding, bias=bias, indice_key=indice_key+"_2", conv_type= conv_types[1], pruning_ratio=pruning_ratio,
                  point_cloud_range=point_cloud_range, voxel_size=voxel_size, algo=algo, pruning_mode=pruning_mode
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, batch_dict):
        identity = x
        if isinstance(self.conv1, (SpatialPrunedSubmConvBlock,)) or isinstance(self.conv1, (SpatialPrunedConvDownsample,)):
            out, batch_dict = self.conv1(x, batch_dict)
        else:
            out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        if isinstance(self.conv2, (SpatialPrunedSubmConvBlock,)) or isinstance(self.conv2, (SpatialPrunedConvDownsample,)):
            out, batch_dict = self.conv2(out, batch_dict)
        else:
            out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out, batch_dict


class VoxelPruningResBackBone8x(nn.Module):
    downsample_type = ["spconv", "spconv", "spconv"]
    downsample_pruning_ratio = [0.5, 0.5, 0.5]
    conv_types = [[["subm", "subm"], ["subm", "subm"]], [["subm", "subm"], ["subm", "subm"]], [["subm", "subm"], ["subm", "subm"]], [["subm", "subm"], ["subm", "subm"]]]
    pruning_ratio = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = PostActBlock

        self.point_cloud_range = model_cfg.get("POINT_CLOUD_RANGE", [0, -40, -3, 70.4, 40, 1])
        self.voxel_size = model_cfg.get("VOXEL_SIZE", [0.1, 0.05, 0.05])
        self.use_native_algo = model_cfg.get("USE_NATIVE_ALGO", True)
        self.pruning_mode = model_cfg.get("PRUNING_MODE", "topk")
        pruning_ratio_cus = model_cfg.get("PRUNING_RATIO", None)
        self.downsample_pruning_mode = model_cfg.get("DOWNSAMPLE_PRUNING_MODE", "topk")
        downsample_pruning_ratio_cus = model_cfg.get("DOWNSAMPLE_PRUNING_RATIO", None)

        if self.use_native_algo:
            self.conv_algo = ConvAlgo.Native
        else:
            self.conv_algo = None
        
        if downsample_pruning_ratio_cus is not None:
            self.downsample_pruning_ratio = downsample_pruning_ratio_cus
        if pruning_ratio_cus is not None:
            self.pruning_ratio = pruning_ratio_cus

        self.conv1 = SparseSequentialBatchdict(
            SparseBasicBlock(16, 16, voxel_stride=1, norm_fn=norm_fn, padding=1, indice_key='res1_1', conv_types= self.conv_types[0][0], pruning_ratio=self.pruning_ratio[0][0],
                  point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, algo=self.conv_algo, pruning_mode=self.pruning_mode),
            SparseBasicBlock(16, 16, voxel_stride=1, norm_fn=norm_fn, padding=1, indice_key='res1_2' if self.conv_types[0][1]==self.conv_types[0][0] else 'res1_2', conv_types= self.conv_types[0][1], pruning_ratio=self.pruning_ratio[0][1],
                  point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, algo=self.conv_algo, pruning_mode=self.pruning_mode),
        )

        self.conv2 = SparseSequentialBatchdict(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, voxel_stride=2, padding=1, indice_key='spconv2', conv_type=self.downsample_type[0], pruning_ratio=self.downsample_pruning_ratio[0],
                  algo=self.conv_algo, pruning_mode=self.pruning_mode, downsample_pruning_mode=self.downsample_pruning_mode, point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size),
            SparseBasicBlock(32, 32, voxel_stride=2, norm_fn=norm_fn, padding=1, indice_key='res2_1', conv_types= self.conv_types[1][0], pruning_ratio=self.pruning_ratio[1][0],
                  point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, algo=self.conv_algo, pruning_mode=self.pruning_mode),
            SparseBasicBlock(32, 32, voxel_stride=2, norm_fn=norm_fn, padding=1, indice_key='res2_2', conv_types= self.conv_types[1][1], pruning_ratio=self.pruning_ratio[1][1],
                  point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, algo=self.conv_algo, pruning_mode=self.pruning_mode),
        )

        self.conv3 = SparseSequentialBatchdict(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, voxel_stride=4, padding=1, indice_key='spconv3', conv_type=self.downsample_type[1], pruning_ratio=self.downsample_pruning_ratio[1],
                  algo=self.conv_algo, pruning_mode=self.pruning_mode, downsample_pruning_mode=self.downsample_pruning_mode, point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size),
            SparseBasicBlock(64, 64, voxel_stride=4, norm_fn=norm_fn, padding=1, indice_key='res3_1', conv_types= self.conv_types[2][0], pruning_ratio=self.pruning_ratio[2][0],
                  point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, algo=self.conv_algo, pruning_mode=self.pruning_mode),
            SparseBasicBlock(64, 64, voxel_stride=4, norm_fn=norm_fn, padding=1, indice_key='res3_2', conv_types= self.conv_types[2][1], pruning_ratio=self.pruning_ratio[2][1],
                  point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, algo=self.conv_algo, pruning_mode=self.pruning_mode),
        )

        self.conv4 = SparseSequentialBatchdict(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, voxel_stride=8, padding=(0, 1, 1), indice_key='spconv4', conv_type=self.downsample_type[2], pruning_ratio=self.downsample_pruning_ratio[2],
                  algo=self.conv_algo, pruning_mode=self.pruning_mode, downsample_pruning_mode=self.downsample_pruning_mode, point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size),
            SparseBasicBlock(128, 128, voxel_stride=8, norm_fn=norm_fn, padding=1, indice_key='res4_1', conv_types= self.conv_types[3][0], pruning_ratio=self.pruning_ratio[3][0],
                  point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, algo=self.conv_algo, pruning_mode=self.pruning_mode),
            SparseBasicBlock(128, 128, voxel_stride=8, norm_fn=norm_fn, padding=1, indice_key='res4_2', conv_types= self.conv_types[3][1], pruning_ratio=self.pruning_ratio[3][1],
                  point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, algo=self.conv_algo, pruning_mode=self.pruning_mode),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)


        x_conv1, batch_dict = self.conv1(x, batch_dict)
        x_conv2, batch_dict = self.conv2(x_conv1, batch_dict)
        x_conv3, batch_dict = self.conv3(x_conv2, batch_dict)
        x_conv4, batch_dict = self.conv4(x_conv3, batch_dict)

        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict



class VoxelPruningBackBone8x(nn.Module):
    downsample_type = ["spconv", "spconv", "spconv"]
    downsample_pruning_ratio = [0.5, 0.5, 0.5]
    conv_types = [["subm", "subm"], ["subm", "subm"], ["subm", "subm"], ["subm", "subm"]]
    pruning_ratio = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = PostActBlock
        self.point_cloud_range = model_cfg.get("POINT_CLOUD_RANGE", [-3, -40, 0, 1, 40, 70.4])
        self.voxel_size = model_cfg.get("VOXEL_SIZE", [0.1, 0.05, 0.05])
        self.use_native_algo = model_cfg.get("USE_NATIVE_ALGO", True)
        self.pruning_mode = model_cfg.get("PRUNING_MODE", "topk")
        pruning_ratio_cus = model_cfg.get("PRUNING_RATIO", None)
        self.downsample_pruning_mode = model_cfg.get("DOWNSAMPLE_PRUNING_MODE", "thre")
        downsample_pruning_ratio_cus = model_cfg.get("DOWNSAMPLE_PRUNING_RATIO", None)
        
        if self.use_native_algo:
            self.conv_algo = ConvAlgo.Native
        else:
            self.conv_algo = None
        
        if downsample_pruning_ratio_cus is not None:
            self.downsample_pruning_ratio = downsample_pruning_ratio_cus
        if pruning_ratio_cus is not None:
            self.pruning_ratio = pruning_ratio_cus
        
        self.conv1 = SparseSequentialBatchdict(
            # block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
            block(16, 16, 3, voxel_stride=1, norm_fn=norm_fn, padding=1, indice_key='subm1_1', conv_type= self.conv_types[0][0], pruning_ratio=self.pruning_ratio[0][0],
                  point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, algo=self.conv_algo, pruning_mode=self.pruning_mode),
        )

        self.conv2 = SparseSequentialBatchdict(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type=self.downsample_type[0], pruning_ratio=self.downsample_pruning_ratio[0],
                  algo=self.conv_algo, pruning_mode=self.pruning_mode, downsample_pruning_mode=self.downsample_pruning_mode, point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size),
            block(32, 32, 3, voxel_stride=2, norm_fn=norm_fn, padding=1, indice_key='subm2_1', conv_type= self.conv_types[1][0], pruning_ratio=self.pruning_ratio[1][0],
                  point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, algo=self.conv_algo, pruning_mode=self.pruning_mode),
            block(32, 32, 3, voxel_stride=2, norm_fn=norm_fn, padding=1, indice_key='subm2_2', conv_type= self.conv_types[1][1], pruning_ratio=self.pruning_ratio[1][1],
                  point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, algo=self.conv_algo, pruning_mode=self.pruning_mode),
        )

        self.conv3 = SparseSequentialBatchdict(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type=self.downsample_type[1], pruning_ratio=self.downsample_pruning_ratio[1],
                  algo=self.conv_algo, pruning_mode=self.pruning_mode, downsample_pruning_mode=self.downsample_pruning_mode, point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size),
            block(64, 64, 3, voxel_stride=4, norm_fn=norm_fn, padding=1, indice_key='subm3_1', conv_type= self.conv_types[2][0], pruning_ratio=self.pruning_ratio[2][0],
                  point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, algo=self.conv_algo, pruning_mode=self.pruning_mode),
            block(64, 64, 3, voxel_stride=4, norm_fn=norm_fn, padding=1, indice_key='subm3_2', conv_type= self.conv_types[2][1], pruning_ratio=self.pruning_ratio[2][1],
                  point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, algo=self.conv_algo, pruning_mode=self.pruning_mode),
        )

        self.conv4 = SparseSequentialBatchdict(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type=self.downsample_type[2], pruning_ratio=self.downsample_pruning_ratio[2],
                  algo=self.conv_algo, pruning_mode=self.pruning_mode, downsample_pruning_mode=self.downsample_pruning_mode, point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size),
            block(64, 64, 3, voxel_stride=8, norm_fn=norm_fn, padding=1, indice_key='subm4_1', conv_type= self.conv_types[3][0], pruning_ratio=self.pruning_ratio[3][0],
                  point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, algo=self.conv_algo, pruning_mode=self.pruning_mode),
            block(64, 64, 3, voxel_stride=8, norm_fn=norm_fn, padding=1, indice_key='subm4_2', conv_type= self.conv_types[3][1], pruning_ratio=self.pruning_ratio[3][1],
                  point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, algo=self.conv_algo, pruning_mode=self.pruning_mode),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        # print("cur_epoch keys:", batch_dict["cur_epoch"])
        # print("total_epoch keys:", batch_dict["total_epochs"])
        # assert False
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        # caluate loss
        batch_dict['loss_box_of_pts'] = 0
        batch_dict['l1_loss'] = 0
        batch_dict['loss_box_of_pts_sprs'] = 0
        batch_dict["3dbackbone_flops"] = 0
        
        
        x = self.conv_input(input_sp_tensor)

        x_conv1, batch_dict = self.conv1(x, batch_dict)
        x_conv2, batch_dict = self.conv2(x_conv1, batch_dict)
        x_conv3, batch_dict = self.conv3(x_conv2, batch_dict)
        x_conv4, batch_dict = self.conv4(x_conv3, batch_dict)
        
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        
        return batch_dict



class VoxelPruningResBackBone8x_SPSS_SPRS(VoxelPruningResBackBone8x):
    conv_types = [[["spss", "spss"], ["spss", "spss"]], [["spss", "spss"], ["spss", "spss"]], [["spss", "spss"], ["spss", "spss"]], [["spss", "spss"], ["spss", "spss"]]]
    downsample_type = ["sprs", "sprs", "sprs"]

class VoxelPruningBackBone8x_SPSS_SPRS(VoxelPruningBackBone8x):
    conv_types = [["spss", "spss"], ["spss", "spss"], ["spss", "spss"], ["spss", "spss"]]
    downsample_type = ["sprs", "sprs", "sprs"]
