
import logging
import numpy as np
import torch
from torchvision import transforms

import utils.data_handling as dh
import utils.unet as unet
import utils.util as util


def main(img_path, gt_path, log_path, train_ids, valid_ids, seed, epochs,
         batch_size, classes, weighting):
    ''' Train model '''

    util.enable_logging(log_path, 'train.log')
    rng = np.random.default_rng(seed)
    train_set = dh.MultibandDataset(img_path, classes, train_ids,
                                    gt_path=gt_path,
                                    transform=transforms.Compose([
                                            dh.RandomFlip(rng),
                                            dh.RandomRotate(rng),
                                            dh.ToTensor(),
                                            dh.ToOnehotGaussianBlur(7,
                                                                    classes,
                                                                    False)]))
    valid_set = dh.MultibandDataset(img_path, classes, valid_ids,
                                    gt_path=gt_path,
                                    transform=transforms.Compose([
                                            dh.ToTensor(),
                                            dh.ToOnehotGaussianBlur(7,
                                                                    classes,
                                                                    False)]))
    torch.manual_seed(seed)
    train_it = torch.utils.data.DataLoader(
                                    train_set, shuffle=True,
                                    batch_size=batch_size, num_workers=0,
                                    generator=torch.Generator('cuda')
                                                   .manual_seed(seed))
    valid_it = torch.utils.data.DataLoader(
                                    valid_set, shuffle=True,
                                    batch_size=batch_size, num_workers=0,
                                    generator=torch.Generator('cuda')
                                                   .manual_seed(seed))

    model = unet.UNet(len(img_path),
                      dh.MultibandDataset.parse_classes(classes))
    logging.info('Start training')
    model.fit(train_it, valid_it, epochs, log_path,
              train_set.infer_weights(weighting))
    logging.info('End training')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                        description='Train Model',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-I', '--img_path', action='append', help='Add path '
                        'to input images')
    parser.add_argument('gt_path', help='Path to groundtruth image folder')
    parser.add_argument('log_path', help='Path to folder where logging data '
                        'will be stored')
    parser.add_argument('train_ids', help='File containing image names for '
                        'training')
    parser.add_argument('valid_ids', help='File containing image names for '
                        'validation')
    parser.add_argument('--seed', help='Random seed', default=None, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', help='Number of patches per batch',
                        type=int, default=4)
    parser.add_argument('--classes', help='List of class labels in ground '
                        'truth - order needs to correspond to weighting order',
                        default='0,1,2')
    parser.add_argument('--weighting', help='Configure class weights - can be '
                        '"mfb", "none", or defined weight string, '
                        'e.g., "0.1,1,1"', default='mfb')

    args = vars(parser.parse_args())
    main(**args)
