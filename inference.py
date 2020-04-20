import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import skimage.io
import argparse
import numpy as np
import time
import os

import nets
import dataloader
from dataloader import transforms
from utils import utils
from utils.file_io import write_pfm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='test', type=str,
                    help='Validation mode on small subset or test mode on full test data')

# Training data
parser.add_argument('--data_dir', default='data/SceneFlow',
                    type=str, help='Training dataset')
parser.add_argument('--dataset_name', default='SceneFlow', type=str, help='Dataset name')

parser.add_argument('--batch_size', default=1, type=int, help='Batch size for inference')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for data loading')
parser.add_argument('--img_height', default=576, type=int, help='Image height for inference')
parser.add_argument('--img_width', default=960, type=int, help='Image width for inference')

# Model
parser.add_argument('--seed', default=326, type=int, help='Random seed for reproducibility')
parser.add_argument('--output_dir', default='output', type=str,
                    help='Directory to save inference results')
parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')

# AANet
parser.add_argument('--feature_type', default='aanet', type=str, help='Type of feature extractor')
parser.add_argument('--no_feature_mdconv', action='store_true', help='Whether to use mdconv for feature extraction')
parser.add_argument('--feature_pyramid', action='store_true', help='Use pyramid feature')
parser.add_argument('--feature_pyramid_network', action='store_true', help='Use FPN')
parser.add_argument('--feature_similarity', default='correlation', type=str,
                    help='Similarity measure for matching cost')
parser.add_argument('--num_downsample', default=2, type=int, help='Number of downsample layer for feature extraction')
parser.add_argument('--aggregation_type', default='adaptive', type=str, help='Type of cost aggregation')
parser.add_argument('--num_scales', default=3, type=int, help='Number of stages when using parallel aggregation')
parser.add_argument('--num_fusions', default=6, type=int, help='Number of multi-scale fusions when using parallel'
                                                               'aggragetion')
parser.add_argument('--num_stage_blocks', default=1, type=int, help='Number of deform blocks for ISA')
parser.add_argument('--num_deform_blocks', default=3, type=int, help='Number of DeformBlocks for aggregation')
parser.add_argument('--no_intermediate_supervision', action='store_true',
                    help='Whether to add intermediate supervision')
parser.add_argument('--deformable_groups', default=2, type=int, help='Number of deformable groups')
parser.add_argument('--mdconv_dilation', default=2, type=int, help='Dilation rate for deformable conv')
parser.add_argument('--refinement_type', default='stereodrnet', help='Type of refinement module')

parser.add_argument('--pretrained_aanet', default=None, type=str, help='Pretrained network')

parser.add_argument('--save_type', default='png', choices=['pfm', 'png', 'npy'], help='Save file type')
parser.add_argument('--visualize', action='store_true', help='Visualize disparity map')

# Log
parser.add_argument('--count_time', action='store_true', help='Inference on a subset for time counting only')
parser.add_argument('--num_images', default=100, type=int, help='Number of images for inference')

args = parser.parse_args()

model_name = os.path.basename(args.pretrained_aanet)[:-4]
model_dir = os.path.basename(os.path.dirname(args.pretrained_aanet))
args.output_dir = os.path.join(args.output_dir, model_dir + '-' + model_name)

utils.check_path(args.output_dir)
utils.save_command(args.output_dir)


def main():
    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test loader
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    test_data = dataloader.StereoDataset(data_dir=args.data_dir,
                                         dataset_name=args.dataset_name,
                                         mode=args.mode,
                                         save_filename=True,
                                         transform=test_transform)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)

    aanet = nets.AANet(args.max_disp,
                       num_downsample=args.num_downsample,
                       feature_type=args.feature_type,
                       no_feature_mdconv=args.no_feature_mdconv,
                       feature_pyramid=args.feature_pyramid,
                       feature_pyramid_network=args.feature_pyramid_network,
                       feature_similarity=args.feature_similarity,
                       aggregation_type=args.aggregation_type,
                       num_scales=args.num_scales,
                       num_fusions=args.num_fusions,
                       num_stage_blocks=args.num_stage_blocks,
                       num_deform_blocks=args.num_deform_blocks,
                       no_intermediate_supervision=args.no_intermediate_supervision,
                       refinement_type=args.refinement_type,
                       mdconv_dilation=args.mdconv_dilation,
                       deformable_groups=args.deformable_groups).to(device)

    # print(aanet)

    if os.path.exists(args.pretrained_aanet):
        print('=> Loading pretrained AANet:', args.pretrained_aanet)
        utils.load_pretrained_net(aanet, args.pretrained_aanet, no_strict=True)
    else:
        print('=> Using random initialization')

    # Save parameters
    num_params = utils.count_parameters(aanet)
    print('=> Number of trainable parameters: %d' % num_params)

    if torch.cuda.device_count() > 1:
        print('=> Use %d GPUs' % torch.cuda.device_count())
        aanet = torch.nn.DataParallel(aanet)

    # Inference
    aanet.eval()

    inference_time = 0
    num_imgs = 0

    num_samples = len(test_loader)
    print('=> %d samples found in the test set' % num_samples)

    for i, sample in enumerate(test_loader):
        if args.count_time and i == args.num_images:  # testing time only
            break

        if i % 100 == 0:
            print('=> Inferencing %d/%d' % (i, num_samples))

        left = sample['left'].to(device)  # [B, 3, H, W]
        right = sample['right'].to(device)

        # Pad
        ori_height, ori_width = left.size()[2:]
        if ori_height < args.img_height or ori_width < args.img_width:
            top_pad = args.img_height - ori_height
            right_pad = args.img_width - ori_width

            # Pad size: (left_pad, right_pad, top_pad, bottom_pad)
            left = F.pad(left, (0, right_pad, top_pad, 0))
            right = F.pad(right, (0, right_pad, top_pad, 0))

        # Warpup
        if i == 0 and args.count_time:
            with torch.no_grad():
                for _ in range(10):
                    aanet(left, right)

        num_imgs += left.size(0)

        with torch.no_grad():
            time_start = time.perf_counter()
            pred_disp = aanet(left, right)[-1]  # [B, H, W]
            inference_time += time.perf_counter() - time_start

        if pred_disp.size(-1) < left.size(-1):
            pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
            pred_disp = F.interpolate(pred_disp, (left.size(-2), left.size(-1)),
                                      mode='bilinear') * (left.size(-1) / pred_disp.size(-1))
            pred_disp = pred_disp.squeeze(1)  # [B, H, W]

        # Crop
        if ori_height < args.img_height or ori_width < args.img_width:
            if right_pad != 0:
                pred_disp = pred_disp[:, top_pad:, :-right_pad]
            else:
                pred_disp = pred_disp[:, top_pad:]

        for b in range(pred_disp.size(0)):
            disp = pred_disp[b].detach().cpu().numpy()  # [H, W]
            save_name = sample['left_name'][b]
            save_name = os.path.join(args.output_dir, save_name)
            utils.check_path(os.path.dirname(save_name))
            if not args.count_time:
                if args.save_type == 'pfm':
                    if args.visualize:
                        skimage.io.imsave(save_name, (disp * 256.).astype(np.uint16))

                    save_name = save_name[:-3] + 'pfm'
                    write_pfm(save_name, disp)
                elif args.save_type == 'npy':
                    save_name = save_name[:-3] + 'npy'
                    np.save(save_name, disp)
                else:
                    skimage.io.imsave(save_name, (disp * 256.).astype(np.uint16))

    print('=> Mean inference time for %d images: %.3fs' % (num_imgs, inference_time / num_imgs))


if __name__ == '__main__':
    main()
