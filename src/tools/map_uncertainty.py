import os

import logging
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torchvision import transforms
from tqdm import tqdm

import utils.util
import utils.unet as unet
import utils.data_handling as dh


def combine_images(input_img, prediction, uncertainty, out_path,
                   class_map):
    '''Combine input image, prediction and uncertainty into one uncertainty map

    Parameters
    ----------
    input_img : Input image map of shape (height, width)
    prediction : Class label prediction of shape (height, width)
    uncertainty : Uncertainty map of shape (height, width)
    out_path : File to write uncertainty map to
    class_map : Map classes found in prediction to color values

    '''
    img = input_img.numpy()
    img = img.reshape((*img.shape, 1))
    img = np.repeat(img, 4, axis=2)

    for cls in class_map:
        if class_map[cls] is None:
            continue
        img[prediction == cls] = class_map[cls]

    uncert = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) -
                                                    np.min(uncertainty))
    alphas = 1 - uncert
    alphas = np.clip(alphas, 0.4, 1)
    img[:, :, 3] = alphas

    plt.imsave(out_path, img)


def perform_mc_dropout(model, predicted, batch, batch_size, class_map,
                       samples, out_path):
    '''Perform MC Dropout for given image and save results

    Parameters
    ----------
    model : Pretrained model
    predicted : Predicted image
    batch : Image to perform uncertainty quantification for
    batch_size : Maximum batch size to process at once
    class_map : Map classes found in prediction to color values
    samples : Number of Monte Carlo samples
    out_path : Folder to write uncertainty maps to

    '''
    img_id = batch['id'][0]
    start = time.time()
    pred_entropy, mutual_info = model.mc_dropout(batch, samples,
                                                 batch_size)
    end = time.time()
    logging.info(f'mc,{img_id},{end-start}')

    np.savez_compressed(os.path.join(out_path, f'mc_en_{img_id}.npz'),
                        uq=pred_entropy)
    np.savez_compressed(os.path.join(out_path, f'mc_mi_{img_id}.npz'),
                        uq=mutual_info)
    plt.imsave(os.path.join(out_path, f'entropy_{img_id}.png'),
               pred_entropy, cmap='Reds')
    plt.imsave(os.path.join(out_path, f'mutual_{img_id}.png'),
               mutual_info, cmap='Reds')
    combine_images(batch['inputs'][0, 0, :, :], predicted, pred_entropy,
                   os.path.join(out_path, f'entropy_comb_{img_id}.png'),
                   class_map)
    combine_images(batch['inputs'][0, 0, :, :], predicted, mutual_info,
                   os.path.join(out_path, f'mutual_comb_{img_id}.png'),
                   class_map)


def calibrate_model(model, classes, img_path, gt_path, calibration_ids,
                    batch_size, model_path):
    '''Calibrate model or load calibration scores from file

    Parameters
    ----------
    model : Pretrained model
    classes : List of class labels
    img_path : List of folders containing input images
    gt_path : Folder containing ground truth images
    calibration_ids : File containing image names to be used for calibration
    batch_size : Maximum batch size to process at once
    model_path : Path to the model - used to store and load calibration scores
    decoder_layer_idx : Index into decoder path at which to split the

    '''
    # check for calibration path
    base_path = os.path.dirname(model_path)
    calibration_file = os.path.join(base_path, model.calibration_name())

    if os.path.isfile(calibration_file):
        logging.warning('Loading scores from existing file: %s',
                        calibration_file)
        model.load_calibration_scores(calibration_file)

    else:
        cal_set = dh.MultibandDataset(img_path, classes, calibration_ids,
                                      gt_path=gt_path,
                                      transform=transforms.Compose([
                                            dh.ToOnehot(classes),
                                            dh.ToTensor()]))
        # ensure that only full batches are submitted for calibration
        cal_it = torch.utils.data.DataLoader(cal_set, batch_size=batch_size,
                                             drop_last=True)
        start = time.time()
        model.calibrate(cal_it)
        end = time.time()
        logging.info(f'{model.log_name()}-calibration,{end-start}')
        model.save_calibration_scores(calibration_file)


def perform_fcp(model, predicted, batch, significance, class_map, out_path):
    '''Perform FCP for given image and save results

    Parameters
    ----------
    model : Calibrated FCP model
    predicted : Predicted image
    batch : Image to perform uncertainty quantification for
    significance : Significance level
    class_map : Map classes found in prediction to color values
    out_path : Folder to write uncertainty maps to

    '''
    img_id = batch['id'][0]
    inputs = batch['inputs']
    _, _, height, width = inputs.shape

    start = time.time()
    intervals = model.predict(inputs, significance)[0]
    end = time.time()
    logging.info(f'fcp,{img_id},{end-start}')

    uncertainty = intervals[:, 1] - intervals[:, 0]
    uncertainty = uncertainty.reshape((len(class_map), height, width))
    uq = utils.util.select_from_index(uncertainty, predicted)
    np.savez_compressed(os.path.join(out_path, f'fcp_{img_id}.npz'),
                        uq=uq, intervals=intervals)
    plt.imsave(os.path.join(out_path, f'fcp_{img_id}_pred.png'),
               uq, cmap='Reds')

    for cls in class_map:
        plt.imsave(os.path.join(out_path, f'fcp_{img_id}_{cls}.png'),
                   uncertainty[cls], cmap='Reds')
        combine_images(inputs[0, 0, :, :], predicted, uncertainty[cls],
                       os.path.join(out_path, f'fcp_comb_{img_id}_{cls}.png'),
                       class_map)


def perform_cr(model, predicted, batch, significance, out_path):
    '''Perform conformal regression on the given image and store the intervals
    and the mean prediction

    Parameters
    ----------
    model : Calibrated CR model
    predicted : Predicted image
    batch : Image to perform uncertainty quantification for
    significance : Significance level
    out_path : Folder to write uncertainty maps to

    '''
    img_id = batch['id'][0]
    inputs = batch['inputs']

    start = time.time()
    uncond_intervals, cond_intervals, means = model.predict(inputs,
                                                            significance)
    end = time.time()
    logging.info(f'cr,{img_id},{end-start}')

    # unconditional results
    uncertainty = uncond_intervals[:, :, :, 1] - uncond_intervals[:, :, :, 0]
    uq = utils.util.select_from_index(uncertainty, predicted)
    np.savez_compressed(os.path.join(out_path, f'cr_{img_id}.npz'),
                        uq=uq, intervals=uncond_intervals, means=means)
    plt.imsave(os.path.join(out_path, f'cr_{img_id}_pred.png'),
               uq, cmap='Reds')

    # conditional results
    uncertainty = cond_intervals[:, :, :, 1] - cond_intervals[:, :, :, 0]
    uq = utils.util.select_from_index(uncertainty, predicted)
    np.savez_compressed(os.path.join(out_path, f'cr_cond_{img_id}.npz'),
                        uq=uq, intervals=cond_intervals, means=means)
    plt.imsave(os.path.join(out_path, f'cr_cond_{img_id}_pred.png'),
               uq, cmap='Reds')


def main(img_path, model_path, out_path, selected_ids, seed, classes,
         mc_dropout, samples, batch_size, fcp, calibration_ids, gt_path,
         significance, decoder_layer_idx, cr, cr_samples):
    '''Generate uncertainty maps for given patches'''

    # setup logging
    utils.util.enable_logging(out_path, 'uncert.log')

    class_map = {0: None, 1: (0, 0, 1, 0), 2: (0, 1, 0, 0)}
    torch.manual_seed(seed)
    # load data
    selected_set = dh.MultibandDataset(img_path, classes, selected_ids,
                                       transform=transforms.Compose([
                                                            dh.ToTensor()]))
    selected_it = torch.utils.data.DataLoader(selected_set, batch_size=1)

    # load model
    model = unet.UNet(len(img_path),
                      dh.MultibandDataset.parse_classes(classes))
    model.load(model_path)

    if fcp:
        fcp_model = unet.ConformalUNet(model, decoder_layer_idx)
        calibrate_model(fcp_model, classes, img_path, gt_path,
                        calibration_ids, batch_size, model_path)
    if cr:
        cr_model = unet.ConformalRegressor(model, cr_samples, batch_size)
        calibrate_model(cr_model, classes, img_path, gt_path,
                        calibration_ids, batch_size, model_path)

    # process images
    for batch in tqdm(selected_it):
        img_id = batch['id'][0]
        predicted = model.proba(batch)[0]
        np.savez_compressed(os.path.join(out_path, f'pred_{img_id}.npz'),
                            predicted=predicted)
        predicted = predicted.argmax(axis=0)
        plt.imsave(os.path.join(out_path, f'pred_{img_id}.png'),
                   predicted, cmap='Reds')

        if mc_dropout:
            perform_mc_dropout(model, predicted, batch, batch_size, class_map,
                               samples, out_path)

        if fcp:
            perform_fcp(fcp_model, predicted, batch, significance, class_map,
                        out_path)

        if cr:
            perform_cr(cr_model, predicted, batch, significance, out_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                        description='Generate uncertainty maps for given '
                        'patches using the given model',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-I', '--img_path', action='append', help='Add path '
                        'to input images')
    parser.add_argument('model_path', help='Path to pre-trained model')
    parser.add_argument('out_path', help='Path to folder where uncertainty '
                        'maps will be stored')
    parser.add_argument('selected_ids', help='File containing image names for '
                        'generating uncertainty maps')
    parser.add_argument('--seed', help='Random seed', default=None, type=int)
    parser.add_argument('--classes', help='List of class labels in ground '
                        'truth - order needs to correspond to weighting order',
                        default='0,1,2')
    parser.add_argument('--batch_size', help='Number of patches per batch',
                        type=int, default=16)

    mc_group = parser.add_argument_group('MC Droput Parameters', 'Configure '
                                         'MC Dropout')
    mc_group.add_argument('--mc_dropout', action='store_true', help='Perform '
                          'MC dropout')
    mc_group.add_argument('--samples', help='Number of samples to generate '
                          'for estimating uncertainty',
                          default=1000, type=int)

    conf_group = parser.add_argument_group('Conformal Parameters', 'Configure '
                                           'FCP and CR')
    conf_group.add_argument('--calibration_ids', help='File containing image '
                            'names to be used for calibration')
    conf_group.add_argument('--gt_path', help='Folder containing ground truth '
                            'images')
    conf_group.add_argument('--significance', help='Required certainty level',
                            type=float, default=0.1)

    fcp_group = parser.add_argument_group('FCP Parameters', 'Configure FCP')
    fcp_group.add_argument('--fcp', action='store_true', help='Perform FCP')
    fcp_group.add_argument('--decoder_layer_idx', help='Indicate at which '
                           'decoder layer the features should be extracted',
                           choices=[0, 1, 2, 3, 4], type=int)

    cr_group = parser.add_argument_group('Conformal Regressor Parameters',
                                         'Configure CR')
    cr_group.add_argument('--cr', action='store_true', help='Perform CR')
    cr_group.add_argument('--cr_samples', help='Number of samples to generate '
                          'for estimating difficulty',
                          default=100, type=int)

    args = vars(parser.parse_args())
    main(**args)
