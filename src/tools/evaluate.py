
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm

import utils.unet as unet
import utils.data_handling as dh


def calculate_f_measure(tp, fp, fn):
    '''Calculate the F1-score

    Parameters
    ----------
    tp : Number of true positives
    fp : Number of false positives
    fn : Number of false negatives

    Returns
    -------
    F1-score

    '''
    return (2*tp) / (2*tp + fp + fn)


def calculate_mcc(confusion):
    '''Calculate MCC

    Parameters
    ----------
    confusion : Confusion matrix

    Returns
    -------
    MCC

    '''
    t = np.zeros((len(confusion),))
    p = np.zeros((len(confusion),))
    c = 0
    s = 0

    for i, cls_id in enumerate(confusion):
        t[i] = confusion[cls_id]['tp'] + confusion[cls_id]['fn']
        p[i] = confusion[cls_id]['tp'] + confusion[cls_id]['fp']
        c += confusion[cls_id]['tp']
        if i == 0:
            s = (confusion[cls_id]['tp'] + confusion[cls_id]['fp']
                 + confusion[cls_id]['tn'] + confusion[cls_id]['fn'])

    numerator = c*s - np.dot(t, p)
    denominator = (np.sqrt(s**2 - np.dot(p, p))
                   * np.sqrt(s**2 - np.dot(t, t)))

    return numerator / denominator


def evaluate_image(confusion, results, img_id, predicted, gt_img):
    '''Calculate F1-score and MCC for the given image and update given results
    and confusion matrix structures

    Parameters
    ----------
    confusion : Confusion matrix for all classes accross images
    results : Results table
    img_id : ID of the current image
    predicted : Predicted image [width, height]
    gt_img : Ground truth image [width, height]

    '''
    results['img'].append(img_id)
    tmp_conf = {f: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for f in confusion}
    for cls_id in confusion:
        cls_pred = predicted == cls_id
        cls_gt = gt_img == cls_id

        cls_tp = np.sum(cls_gt & cls_pred)
        cls_fp = np.sum((1 - cls_gt) & cls_pred)
        cls_tn = np.sum((1 - cls_gt) & (1 - cls_pred))
        cls_fn = np.sum(cls_gt & (1 - cls_pred))

        results[f'f_{cls_id}'].append(calculate_f_measure(cls_tp, cls_fp,
                                                          cls_fn))
        confusion[cls_id]['tp'] += cls_tp
        confusion[cls_id]['fp'] += cls_fp
        confusion[cls_id]['tn'] += cls_tn
        confusion[cls_id]['fn'] += cls_fn

        tmp_conf[cls_id]['tp'] = cls_tp
        tmp_conf[cls_id]['fp'] = cls_fp
        tmp_conf[cls_id]['tn'] = cls_tn
        tmp_conf[cls_id]['fn'] = cls_fn

    results['mcc'].append(calculate_mcc(tmp_conf))


def main(img_path, gt_path, model_path, out_file, selected_ids, classes):
    # load data
    selected_set = dh.MultibandDataset(img_path, classes, selected_ids,
                                       gt_path=gt_path,
                                       transform=transforms.Compose([
                                                            dh.ToTensor()]))
    selected_it = torch.utils.data.DataLoader(selected_set, batch_size=1)

    # load model
    model = unet.UNet(len(img_path),
                      dh.MultibandDataset.parse_classes(classes))
    model.load(model_path)

    classes = dh.MultibandDataset.parse_classes(classes)
    results = {'img': [], 'mcc': []}
    confusion = {f: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for f in classes}
    for cls_id in classes:
        results[f'f_{cls_id}'] = []

    # process images
    for batch in tqdm(selected_it):
        img_id = batch['id'][0]
        predicted = model.predict(batch)[0]
        target = batch['target'][0].numpy()

        evaluate_image(confusion, results, img_id, predicted,
                       target)

    # add summary statistics
    results['img'].append('all')
    for cls_id in confusion:
        results[f'f_{cls_id}'].append(calculate_f_measure(
                                                    confusion[cls_id]['tp'],
                                                    confusion[cls_id]['fp'],
                                                    confusion[cls_id]['fn']))
    results['mcc'].append(calculate_mcc(confusion))

    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                        description='Evaluate trained model on given dataset.',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-I', '--img_path', action='append', help='Add path '
                        'to input images')
    parser.add_argument('gt_path', help='Path to groundtruth image folder')
    parser.add_argument('model_path', help='Path to pre-trained model')
    parser.add_argument('out_file', help='Output CSV file')
    parser.add_argument('selected_ids', help='File containing image names for '
                        'generating uncertainty maps')
    parser.add_argument('--classes', help='List of class labels in ground '
                        'truth - order needs to correspond to weighting order',
                        default='0,1,2')

    args = vars(parser.parse_args())
    main(**args)
