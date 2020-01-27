import h5py
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import sys
import six
import os

from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image

import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


IMAGENET_NORMALIZATION_PARAMS = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


class ImageNetResNetFeature(nn.Module):
    def __init__(self, output_dim="2048"):
        super(ImageNetResNetFeature, self).__init__()
        resnet = models.resnet152(pretrained=True)
        if output_dim == "2048":
            n_layers_to_rm = 1  # remove last fc layer
        elif output_dim == "2048x7x7":
            n_layers_to_rm = 2  # remove last fc layer and its precedent 7x7 avg pooling layer
        else:
            raise ValueError("Wrong value for argument output_dim")
        self.feature = nn.Sequential(*list(resnet.children())[:-n_layers_to_rm])

    def forward(self, x):
        """return: B x 2048 or B x 2048x7x7"""
        return self.feature(x).squeeze()


class ResNetC3FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetC3FeatureExtractor, self).__init__()
        resnet = models.resnet152(pretrained=True)
        component_list = list(resnet.children())[:-3]
        component_list.extend(list(resnet.layer4.children())[:2])
        self.resnet_base = nn.Sequential(*component_list)
        layer4_children = list(resnet.layer4.children())[2]

        # resnet.layer4[2].downsample is None
        self.layer4_head = nn.Sequential(
            layer4_children.conv1,
            layer4_children.bn1,
            layer4_children.relu,
            layer4_children.conv2,
            layer4_children.bn2,
            layer4_children.relu,
        )

    def forward(self, x):
        base_out = self.resnet_base(x)
        c3_feature = self.layer4_head(base_out)
        return c3_feature


def make_image_tensor(image_paths, zoom_out=1):
    tensors = []
    for ele in image_paths:
        image = Image.open(ele).convert('RGB')
        image = imagenet_transform(image)
        image = image.view(1, 3, 224*zoom_out, 224*zoom_out)
        tensors.append(image)
    return torch.cat(tensors, 0)


def get_image_batch_features(image_paths, net, batch_size, zoom_out=1):
    """
    input:
        path to the frames for a single video
    return:
        image features for the frames
    """
    num_batches = int(np.ceil(float(len(image_paths)) / batch_size))
    feature_list = []
    for i in range(num_batches):
        inputs = make_image_tensor(image_paths[i*batch_size:(i+1)*batch_size], zoom_out=zoom_out)
        inputs = inputs.cuda()
        cur_features = net(inputs)
        feature_list.append(cur_features)
    features = torch.cat(feature_list, 0)
    return features.data.cpu().numpy()


def extract_all(feature_path, base_dir, video_name2image_filenames, video_names, net, batch_size,
                zoom_out=1, debug=False):
    """
    Args:
        feature_path: h5py file path to save the features
        base_dir: os.path.join(base_dir, vid_name, image_filename) is the absolute path to the image
        video_name2image_filenames:  dict(), with video names as keys, list of image filenames as values
        video_names:
        net:
        batch_size:
        zoom_out:
        debug:

    Returns:

    """
    feature_h5 = h5py.File(feature_path, "w")

    for i in tqdm(range(len(video_names)), desc="Extracting for videos"):
        cur_vname = video_names[i]
        image_paths = [os.path.join(base_dir, cur_vname, e) for e in video_name2image_filenames[cur_vname]]
        try:
            data_features = get_image_batch_features(image_paths, net, batch_size, zoom_out=zoom_out)
        except Exception as e:
            logger.debug(e)
            continue
        feature_h5.create_dataset(cur_vname, data=data_features, dtype=np.float32)

        if debug:
            logger.info("subdir (key name) {}, feature shape {}".format(cur_vname, data_features.shape))
            break
    feature_h5.close()


def get_image_paths(dir_path, image_filename_pattern="img_{:05d}.jpg", fps=15):
    """each dir contains the same number of flow_x_{:05d}.jpg, flow_y_{:05d}.jpg, img_{:05d}.jpg.
    Index starts at 1, not 0, thus there is no img_00000.jpg, etc.
    """
    num_rgb_images = int(len(os.listdir(dir_path)) / 3)  # must be divisible by 3
    offsets_per_second = np.arange(0, num_rgb_images, fps)  # (0, 30, 15) => [0, 15]
    # original frames are extracted for the following frames, (video fps=30): [1-5], [11-15], [21-25] + 30*n
    offsets_inside_second = [3, 8, 13]  # the middle of every 5 frames., note this is not used for indexing.
    selected_img_indices = np.concatenate(
        [offsets_per_second + e for e in offsets_inside_second]
        , axis=0)
    selected_img_indices = selected_img_indices[selected_img_indices <= num_rgb_images]
    return [image_filename_pattern.format(e) for e in selected_img_indices]


if __name__ == "__main__":
    # settings
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_file", type=str, default=None)
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--feature_type", type=str, default="imagenet2048",
                        choices=["2048", "2048x7x7", "c3"])
    parser.add_argument("--zoom_out", type=int, default=1, help="224 * zoom_out is the input spatial size")
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--bypass_user_input", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.info(vars(args))

    logger.info("[Phase 1] Setup feature extractor.")
    # https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-152-deploy.prototxt
    # see the link above for resnet architectrue, layer_name, etc.
    feature_type = args.feature_type
    if feature_type == "2048":
        extractor = ImageNetResNetFeature(output_dim="2048")
    elif feature_type == "2048x7x7":
        extractor = ImageNetResNetFeature(output_dim="2048x7x7")
    elif feature_type == "c3":
        extractor = ResNetC3FeatureExtractor()
    else:
        raise NotImplementedError("Not supported feature type")

    # Step 2, set experiment settings
    logger.info("[Phase 2] Config settings.")

    if os.path.exists(args.feature_file):
        logger.info("feature_file {} already exists".format(args.feature_file))
        sys.exit(1)

    USE_CUDA = torch.cuda.is_available()
    if not USE_CUDA:
        logger.info("no GPU available")
        sys.exit(1)
    cudnn.benchmark = True

    extractor.cuda()
    extractor.eval()

    zoom_out = args.zoom_out
    # testing
    with torch.no_grad():
        sample_input = torch.randn(args.batch_size, 3, 224 * zoom_out, 224 * zoom_out)
        if USE_CUDA:
            sample_input = sample_input.cuda()
            logger.info(" Extraction on GPU.")
        sample_output1 = extractor(sample_input)

    logger.info(" Input Size is: {}".format(sample_input.shape))
    logger.info(" Feature Size is: {}".format(sample_output1.shape))
    if args.bypass_user_input:
        s = "y"
    else:
        s = six.moves.input("Do you want to proceed (Y/N): ")
    if s.lower() == "y":
        imagenet_transform = transforms.Compose([
                    transforms.Resize((224 * zoom_out, 224 * zoom_out)),
                    transforms.ToTensor(),
                    transforms.Normalize(**IMAGENET_NORMALIZATION_PARAMS),
                ])

        logger.info("[Phase 3] : Feature Extraction")
        sub_dirs = [d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, d))]
        cache_video_name2image_filenames_path = \
            os.path.join(args.cache_dir, "{}_video_name2image_filenames.cache.pt"
                                         .format(os.path.split(args.feature_file)[-1]))
        if os.path.exists(cache_video_name2image_filenames_path):
            logger.info("Loading from cache {}".format(cache_video_name2image_filenames_path))
            video_name2image_filenames = torch.load(cache_video_name2image_filenames_path)
        else:
            logger.info("Cache not found, creating and saving at {}"
                        .format(cache_video_name2image_filenames_path))
            video_name2image_filenames = {
                k: get_image_paths(os.path.join(args.base_dir, k))
                for k in tqdm(sub_dirs, desc="Gathering image paths for each video")
            }
            torch.save(video_name2image_filenames, cache_video_name2image_filenames_path)
        logger.info("video_name2image_filenames len {} keys[:3] {} values [0][:10] {}"
                    .format(len(video_name2image_filenames),
                            list(video_name2image_filenames.keys())[:3],
                            list(video_name2image_filenames.values())[0][:10]))
        with torch.no_grad():
            extract_all(args.feature_file, args.base_dir, video_name2image_filenames, sub_dirs, extractor,
                        args.batch_size, zoom_out=zoom_out, debug=args.debug)
    else:
        logging.info("Aborting")
