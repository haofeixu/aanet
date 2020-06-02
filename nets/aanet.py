import torch.nn as nn
import torch.nn.functional as F

from nets.feature import (StereoNetFeature, PSMNetFeature, GANetFeature, GCNetFeature,
                          FeaturePyrmaid, FeaturePyramidNetwork)
from nets.resnet import AANetFeature
from nets.cost import CostVolume, CostVolumePyramid
from nets.aggregation import (StereoNetAggregation, GCNetAggregation, PSMNetBasicAggregation,
                              PSMNetHGAggregation, AdaptiveAggregation)
from nets.estimation import DisparityEstimation
from nets.refinement import StereoNetRefinement, StereoDRNetRefinement, HourglassRefinement


class AANet(nn.Module):
    def __init__(self, max_disp,
                 num_downsample=2,
                 feature_type='aanet',
                 no_feature_mdconv=False,
                 feature_pyramid=False,
                 feature_pyramid_network=False,
                 feature_similarity='correlation',
                 aggregation_type='adaptive',
                 num_scales=3,
                 num_fusions=6,
                 deformable_groups=2,
                 mdconv_dilation=2,
                 refinement_type='stereodrnet',
                 no_intermediate_supervision=False,
                 num_stage_blocks=1,
                 num_deform_blocks=3):
        super(AANet, self).__init__()

        self.refinement_type = refinement_type
        self.feature_type = feature_type
        self.feature_pyramid = feature_pyramid
        self.feature_pyramid_network = feature_pyramid_network
        self.num_downsample = num_downsample
        self.aggregation_type = aggregation_type
        self.num_scales = num_scales

        # Feature extractor
        if feature_type == 'stereonet':
            self.max_disp = max_disp // (2 ** num_downsample)
            self.num_downsample = num_downsample
            self.feature_extractor = StereoNetFeature(self.num_downsample)
        elif feature_type == 'psmnet':
            self.feature_extractor = PSMNetFeature()
            self.max_disp = max_disp // (2 ** num_downsample)
        elif feature_type == 'gcnet':
            self.feature_extractor = GCNetFeature()
            self.max_disp = max_disp // 2
        elif feature_type == 'ganet':
            self.feature_extractor = GANetFeature(feature_mdconv=(not no_feature_mdconv))
            self.max_disp = max_disp // 3
        elif feature_type == 'aanet':
            self.feature_extractor = AANetFeature(feature_mdconv=(not no_feature_mdconv))
            self.max_disp = max_disp // 3
        else:
            raise NotImplementedError

        if feature_pyramid_network:
            if feature_type == 'aanet':
                in_channels = [32 * 4, 32 * 8, 32 * 16, ]
            else:
                in_channels = [32, 64, 128]
            self.fpn = FeaturePyramidNetwork(in_channels=in_channels,
                                             out_channels=32 * 4)
        elif feature_pyramid:
            self.fpn = FeaturePyrmaid()

        # Cost volume construction
        if feature_type == 'aanet' or feature_pyramid or feature_pyramid_network:
            cost_volume_module = CostVolumePyramid
        else:
            cost_volume_module = CostVolume
        self.cost_volume = cost_volume_module(self.max_disp,
                                              feature_similarity=feature_similarity)

        # Cost aggregation
        max_disp = self.max_disp
        if feature_similarity == 'concat':
            in_channels = 64
        else:
            in_channels = 32  # StereoNet uses feature difference

        if aggregation_type == 'adaptive':
            self.aggregation = AdaptiveAggregation(max_disp=max_disp,
                                                   num_scales=num_scales,
                                                   num_fusions=num_fusions,
                                                   num_stage_blocks=num_stage_blocks,
                                                   num_deform_blocks=num_deform_blocks,
                                                   mdconv_dilation=mdconv_dilation,
                                                   deformable_groups=deformable_groups,
                                                   intermediate_supervision=not no_intermediate_supervision)
        elif aggregation_type == 'psmnet_basic':
            self.aggregation = PSMNetBasicAggregation(max_disp=max_disp)
        elif aggregation_type == 'psmnet_hourglass':
            self.aggregation = PSMNetHGAggregation(max_disp=max_disp)
        elif aggregation_type == 'gcnet':
            self.aggregation = GCNetAggregation()
        elif aggregation_type == 'stereonet':
            self.aggregation = StereoNetAggregation(in_channels=in_channels)
        else:
            raise NotImplementedError

        match_similarity = False if feature_similarity in ['difference', 'concat'] else True

        if 'psmnet' in self.aggregation_type:
            max_disp = self.max_disp * 4  # PSMNet directly upsamples cost volume
            match_similarity = True  # PSMNet learns similarity for concatenation

        # Disparity estimation
        self.disparity_estimation = DisparityEstimation(max_disp, match_similarity)

        # Refinement
        if self.refinement_type is not None and self.refinement_type != 'None':
            if self.refinement_type in ['stereonet', 'stereodrnet', 'hourglass']:
                refine_module_list = nn.ModuleList()
                for i in range(num_downsample):
                    if self.refinement_type == 'stereonet':
                        refine_module_list.append(StereoNetRefinement())
                    elif self.refinement_type == 'stereodrnet':
                        refine_module_list.append(StereoDRNetRefinement())
                    elif self.refinement_type == 'hourglass':
                        refine_module_list.append(HourglassRefinement())
                    else:
                        raise NotImplementedError

                self.refinement = refine_module_list
            else:
                raise NotImplementedError

    def feature_extraction(self, img):
        feature = self.feature_extractor(img)
        if self.feature_pyramid_network or self.feature_pyramid:
            feature = self.fpn(feature)
        return feature

    def cost_volume_construction(self, left_feature, right_feature):
        cost_volume = self.cost_volume(left_feature, right_feature)

        if isinstance(cost_volume, list):
            if self.num_scales == 1:
                cost_volume = [cost_volume[0]]  # ablation purpose for 1 scale only
        elif self.aggregation_type == 'adaptive':
            cost_volume = [cost_volume]
        return cost_volume

    def disparity_computation(self, aggregation):
        if isinstance(aggregation, list):
            disparity_pyramid = []
            length = len(aggregation)  # D/3, D/6, D/12
            for i in range(length):
                disp = self.disparity_estimation(aggregation[length - 1 - i])  # reverse
                disparity_pyramid.append(disp)  # D/12, D/6, D/3
        else:
            disparity = self.disparity_estimation(aggregation)
            disparity_pyramid = [disparity]

        return disparity_pyramid

    def disparity_refinement(self, left_img, right_img, disparity):
        disparity_pyramid = []
        if self.refinement_type is not None and self.refinement_type != 'None':
            if self.refinement_type == 'stereonet':
                for i in range(self.num_downsample):
                    # Hierarchical refinement
                    scale_factor = 1. / pow(2, self.num_downsample - i - 1)
                    if scale_factor == 1.0:
                        curr_left_img = left_img
                        curr_right_img = right_img
                    else:
                        curr_left_img = F.interpolate(left_img,
                                                      scale_factor=scale_factor,
                                                      mode='bilinear', align_corners=False)
                        curr_right_img = F.interpolate(right_img,
                                                       scale_factor=scale_factor,
                                                       mode='bilinear', align_corners=False)
                    inputs = (disparity, curr_left_img, curr_right_img)
                    disparity = self.refinement[i](*inputs)
                    disparity_pyramid.append(disparity)  # [H/2, H]

            elif self.refinement_type in ['stereodrnet', 'hourglass']:
                for i in range(self.num_downsample):
                    scale_factor = 1. / pow(2, self.num_downsample - i - 1)

                    if scale_factor == 1.0:
                        curr_left_img = left_img
                        curr_right_img = right_img
                    else:
                        curr_left_img = F.interpolate(left_img,
                                                      scale_factor=scale_factor,
                                                      mode='bilinear', align_corners=False)
                        curr_right_img = F.interpolate(right_img,
                                                       scale_factor=scale_factor,
                                                       mode='bilinear', align_corners=False)
                    inputs = (disparity, curr_left_img, curr_right_img)
                    disparity = self.refinement[i](*inputs)
                    disparity_pyramid.append(disparity)  # [H/2, H]

            else:
                raise NotImplementedError

        return disparity_pyramid

    def forward(self, left_img, right_img):
        left_feature = self.feature_extraction(left_img)
        right_feature = self.feature_extraction(right_img)
        cost_volume = self.cost_volume_construction(left_feature, right_feature)
        aggregation = self.aggregation(cost_volume)
        disparity_pyramid = self.disparity_computation(aggregation)
        disparity_pyramid += self.disparity_refinement(left_img, right_img,
                                                       disparity_pyramid[-1])

        return disparity_pyramid
