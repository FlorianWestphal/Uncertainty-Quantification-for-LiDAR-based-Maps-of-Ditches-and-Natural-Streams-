import numpy as np
import os
import pandas as pd
import tifffile

import utils.data_handling as dh
import utils.util

UQ = ['prob', 'mc_en', 'mc_mi', 'cr', 'cr_cond', 'fcp', 'oracle']
CONF_UQ = UQ[1:-1]
CM = ['tp', 'fp', 'tn', 'fn']
IMG_WIDTH = 500
IMG_HEIGHT = 500


def calculate_step(results, uq_id, classes, predicted, target, corrected,
                   step_id):
    '''Derive confusion matrix for given step
    Parameters
    ----------
    results : Array to store results in [TYPEs, UQs, STEPs, CLSs, CLSs]
    uq_method : Uncertainty quantification method to analyse
    classes : List of class labels
    predicted : Predicted class probabilities
    target : Tensor of targets
    corrected : Tensor of corrected pixels
    step : Current step

    '''
    for true_cls in classes:
        cls_gt = target == true_cls
        cls_corr = corrected == true_cls
        for pred_cls in classes:
            cls_pred = predicted == pred_cls

            results[0, uq_id, step_id,
                    true_cls, pred_cls] = np.sum(cls_pred & cls_gt)
            results[1, uq_id, step_id,
                    true_cls, pred_cls] = np.sum(cls_pred & cls_gt)
            if true_cls == pred_cls:
                results[1, uq_id, step_id,
                        true_cls, pred_cls] += np.sum(cls_corr)


def step_through(results, uq_id, classes, uqs,
                 predicted, target, steps, uncertainties=None):
    '''Compute confusion matrices for sparsificaiton and correction for each
       step for the given uncertainty quantification method

    Parameters
    ----------
    results : Array to store results in [TYPEs, UQs, STEPs, CLSs, CLSs]
    uq_id : Uncertainty quantification method index to analyse
    classes : List of class labels
    uqs : List of estimated uncertainties
    predicted : Predicted class probabilities
    target : Tensor of targets
    steps : Fractions used in each step
    uncertainties : Array to store uncertainties remaining in each step
                    [UQs, STEPs, VALUES]

    '''
    uqs_idxs = uqs.argsort()
    for step_id, step in enumerate(steps):
        thresh = int(np.floor(step * len(uqs_idxs)))
        if thresh == 0:
            selected = uqs_idxs
            corrected = np.array([])
        else:
            selected = uqs_idxs[:-thresh]
            corr_idxs = uqs_idxs[len(uqs_idxs) - thresh:]
            corrected = target[corr_idxs]

        if uncertainties is not None:
            uncertainties[uq_id, step_id, 0] = np.mean(uqs[selected])
            uncertainties[uq_id, step_id, 1] = np.std(uqs[selected])
        calculate_step(results, uq_id, classes, predicted[selected],
                       target[selected], corrected, step_id)


def calcualte_sparsification(results, indiv_results, uncertainties, uq_id,
                             selected_ids, gt_path, map_path, classes, steps):
    '''Perform sparsification for the given uncertainty quantification method
    or the oracle across all pixels of the selected images

    Parameters
    ----------
    results : Array to store results in [TYPEs, UQs, STEPs, CLSs, CLSs]
    indiv_results : Array to store focused results in
                    [CHNLs, UQs, STEPs, CLSs, CLSs]
    uncertainties : Array to store uncertainties remaining in each step
                    [UQs, STEPs, VALUES]
    uq_id : Uncertainty quantification method index to analyse
    selected_ids : List of images to perform calculation on
    gt_path : Path to the folder with the groundtruth images
    map_path : Path to the precomputed uncertainty maps
    classes : List of class labels
    steps : Fractions to use in each step

    '''
    # lists to store predicted values, gt values and uncertainties across all
    # images
    pred_vector = np.array([])
    target_vector = np.array([])
    uq_vector = np.array([])

    for img_name in selected_ids:
        uq_method = UQ[uq_id]
        predicted = np.load(os.path.join(map_path,
                                         f'pred_{img_name}.npz'))['predicted']
        target = tifffile.imread(os.path.join(gt_path, f'{img_name}.tif'))
        if uq_method == 'oracle':
            # calculate cross-entropy per pixel
            # $\sum_j^k y_k * log(\hat{y}_k)$ -> only the probability of the
            # true class is selected
            # get probabilities for the true classes
            probs = utils.util.select_from_index(predicted, target)
            # for numerical stability
            probs = np.clip(probs, 1e-12, 1-1e-12)
            # compute the negative log of the respective probabilities
            uqs = -np.log(probs)
        elif uq_method == 'prob':
            # select the maximum probability for each pixel
            uqs = utils.util.select_from_index(predicted,
                                               predicted.argmax(axis=0))
            # convert prediction probabilities into uncertainties
            uqs = 1 - uqs
        else:
            uqs = np.load(os.path.join(map_path,
                                       f'{uq_method}_{img_name}.npz'))['uq']

        uqs = uqs.flatten()
        target = target.flatten()
        predicted = predicted.argmax(axis=0).flatten()

        # store all predictions, gt values and uncertainties
        pred_vector = np.concatenate([pred_vector, predicted])
        target_vector = np.concatenate([target_vector, target])
        uq_vector = np.concatenate([uq_vector, uqs])

    # normalize uq values
    uq_min = uq_vector.min()
    uq_max = uq_vector.max()
    uq_vector = (uq_vector - uq_min) / (uq_max - uq_min)
    step_through(results, uq_id, classes, uq_vector, pred_vector,
                 target_vector, steps, uncertainties)

    # get all pixels predicted as stream
    tmp_results = np.zeros(results.shape)
    step_through(tmp_results, uq_id, classes, uq_vector[pred_vector == 2],
                 pred_vector[pred_vector == 2],
                 target_vector[pred_vector == 2], steps)
    # compute confusion matrix without predicted stream pixels and add to all
    # confusion matrices
    # conf = np.zeros((2, 1, 1, 3, 3))
    # calculate_step(conf, 0, classes, pred_vector[pred_vector != 2],
    #                target_vector[pred_vector != 2], [], 0)
    # indiv_results[0, uq_id] = (tmp_results[1, uq_id]
    #                            + conf[0, 0].repeat(len(steps), axis=0))
    indiv_results[0, uq_id] = tmp_results[1, uq_id]

    # get all pixels predicted as ditch
    tmp_results = np.zeros(results.shape)
    step_through(tmp_results, uq_id, classes, uq_vector[pred_vector == 1],
                 pred_vector[pred_vector == 1],
                 target_vector[pred_vector == 1], steps)
    # compute confusion matrix without predicted stream pixels and add to all
    # confusion matrices
    # conf = np.zeros((2, 1, 1, 3, 3))
    # calculate_step(conf, 0, classes, pred_vector[pred_vector != 1],
    #                target_vector[pred_vector != 1], [], 0)
    # indiv_results[1, uq_id] = (tmp_results[1, uq_id]
    #                            + conf[0, 0].repeat(len(steps), axis=0))
    indiv_results[1, uq_id] = tmp_results[1, uq_id]

    return len(uq_vector), np.sum(pred_vector == 2), np.sum(pred_vector == 1)


def compute_fmes(confusion, cls_id):
    '''Compute the F-Measure for the given class based on the confusion matrix

    Parameters
    ----------
    confusion : Confusion matrix [true class, predicted class]
    cls_id : Class used as positive class

    Returns
    -------
    F-Measure value

    '''
    tpos = confusion[cls_id, cls_id]
    # false negatives are all positives, except the true positives
    fneg = confusion[cls_id, :].sum() - tpos
    # false positive are all predicted positives, except the true positives
    fpos = confusion[:, cls_id].sum() - tpos

    return 2*tpos / (2*tpos + fpos + fneg)


def compute_mcc(confusion):
    '''Compute the MCC value for the given confusion matrix

    Parameters
    ----------
    confusion : Confusion matrix [true class, predicted class]

    Returns
    -------
    MCC value

    '''
    epsilon = 1e-5

    # actual occurences of each class
    t = confusion.sum(axis=1)
    # number of times each class was predicted
    p = confusion.sum(axis=0)
    # number of correct predictions accross classes
    c = confusion.trace()
    # number of samples
    s = confusion.sum()

    numerator = c*s - np.dot(t, p)
    denominator = (np.sqrt(s**2 - np.dot(p, p))
                   * np.sqrt(s**2 - np.dot(t, t)))

    return numerator / (denominator + epsilon)


def compute_iam(confusion):
    '''Compute IAM for the given confusion matrix

    E. Mortaz, "Imbalance accuracy metric for model selection in multi-class
    imbalance classification problems", 2020

    Parameters
    ----------
    confusion : Confusion matrix [true class, predicted class]

    Returns
    -------
    IAM value

    '''
    cls_no = confusion.shape[-1]
    classes = range(cls_no)
    sum = 0
    for cls_id in classes:
        idx = [c for c in classes if c != cls_id]
        numerator = confusion[cls_id, cls_id] - max(confusion[cls_id,
                                                              idx].sum(),
                                                    confusion[idx,
                                                              cls_id].sum())
        denominator = max(confusion[cls_id, :].sum(),
                          confusion[:, cls_id].sum())
        if denominator == 0:    # or confusion[cls_id, :].sum() == 0:
            # if the class was neither in the ground truth nor predicted,
            # pretend the class did not exist
            cls_no -= 1
        else:
            sum += numerator/denominator

    # handle the case that the entire confusion matrix is empty
    if cls_no == 0:
        return np.nan

    return 1/cls_no * sum


def compute_perf(total_confusion, classes):
    '''Take matrix containing confusion matrices for all uncertainty
    quantification methods, all patches, all classes, and all sparsification
    steps and compute the MCC and F1 score for all UQ methods, all classes, and
    all sparsification steps - MCC for multi-class and F1 score for single
    class

    Parameters
    ----------
    total_confusion : Confusion matrix [UQ, steps, true class, predicted class]

    Returns
    -------
    F1 scores and MCC values [UQ, steps, classes]

    '''
    results = np.zeros((len(UQ), total_confusion.shape[1], len(classes)+1))
    for uq_i, _ in enumerate(UQ):
        confusion = total_confusion[uq_i]
        # compute MCC over all classes for each step
        for step in range(confusion.shape[0]):
            results[uq_i, step, len(classes)] = compute_mcc(confusion[step])

        # compute F1 score for each class and each step
        for cls_id in classes:
            idx = [c for c in classes if c != cls_id]
            for step in range(confusion.shape[0]):
                conf = confusion[step]
                # derive reduced confusion matrix
                tmp = np.zeros((conf.shape[0]-1, conf.shape[1]-1))
                tmp[0, 0] = conf[cls_id, cls_id]
                tmp[0, 1] = conf[cls_id, idx].sum()
                tmp[1, 0] = conf[idx, cls_id].sum()
                tmp[1, 1] = conf[sorted(idx+idx), idx+idx].sum()
                results[uq_i, step, cls_id] = compute_fmes(tmp, 0)

    return results


def compute_auses(iam_err, steps):
    '''Compute Area under the Sparsification Error Curve using the trapezoid
    approach

    Parameters
    ----------
    iam_err : IAM errors [UQ, steps, classes]
    steps : Fractions used in each step

    Returns
    -------
    AUSEs [UQ, classes]

    '''
    results = np.zeros((iam_err.shape[0], iam_err.shape[-1]))
    for uq_i in range(iam_err.shape[0]):
        area = 0
        for i in range(len(steps) - 2):
            # trapezoid area: (a+b)/2 * h
            h = steps[i+1] - steps[i]
            a = iam_err[uq_i, i]
            b = iam_err[uq_i, i+1]
            area += ((a+b)/2) * h
        results[uq_i] = area

    return results


def derive_class_set(upper, lower, classes):
    '''Create set of class candidates for each pixel based on if their
    probability intervals overlap

    Parameters
    ----------
    upper : Upper probability bound
    lower : Lower probability bound
    classes : List of class labels

    Returns
    -------
    class set [classes, height, width]

    '''
    class_set = np.zeros((len(classes), *upper.shape[1:]), dtype=np.uint8)

    selected_idxs = upper.argmax(axis=0)
    selected_lower = utils.util.select_from_index(lower, selected_idxs)

    for cls_i in range(len(classes)):
        class_set[cls_i] = (upper[cls_i] >= selected_lower).astype(np.uint8)

    return class_set


def assess_correctness(tmp_acc, class_set, target, uq_methods, classes, uq_i):
    '''Aggregate pixel correctness based on given class set and target

    Parameters
    ----------
    tmp_acc : Array to store results in [UQs+1, classes+1]
    class_set : Indication of which classes were selected based on intervals
    target : Tensor of targets
    classes : List of class labels
    uq_i : Uncertainty quantification index

    '''
    for cls_i, cls_id in enumerate(classes):
        cls_gt = target == cls_id
        # count positives per class - only once
        if uq_i == 0:
            tmp_acc[len(uq_methods), cls_i] += cls_gt.sum()
        # count true positives per class
        tmp_acc[uq_i, cls_i] += np.sum(class_set[cls_i] & cls_gt)


def write_correctness(uq_methods, classes, tmp_acc, out_path,
                      name='uq_correctness.csv'):
    '''Compute and write accuracies to file

    Parameters
    ----------
    uq_methods : Methods to perfor evaluation for
    classes : List of class labels
    tmp_acc : Array to store results in [UQs+1, classes+1]
    out_path : Folder to write results to
    name : CSV file name, optional

    '''
    results = {'class': [*classes, 'all']}

    for uq_i, uq_method in enumerate(uq_methods):
        tmp = []
        for cls_i, _ in enumerate(classes):
            # compute recall per class
            tmp.append(tmp_acc[uq_i, cls_i] / tmp_acc[len(uq_methods), cls_i])
        total = tmp_acc[len(uq_methods), :len(classes)].sum()
        # check in how many cases we got it right vs. total number of instances
        tmp.append(tmp_acc[uq_i, :len(classes)].sum() / total)

        results[uq_method] = tmp

    pd.DataFrame(results).to_csv(os.path.join(out_path, name), index=False)


def assess_band_size(results, classes, uq_i, img_i, intervals):
    '''Estimate average band sizes

    Parameters
    ----------
    results : Array to store results in [UQs, IMGs, CLSs + 1]
    classes : List of class labels
    uq_i : Uncertainty method id
    img_i : Selected image id
    intervals : Probability ranges [CLSs, HGT, WDT]

    '''
    clipped = np.clip(intervals, 0, 1)
    bands = clipped[:, :, :, 1] - clipped[:, :, :, 0]

    results[uq_i, img_i,
            0:len(classes)] = bands.reshape((bands.shape[0], -1)).mean(axis=1)
    results[uq_i, img_i, len(classes)] = bands.mean()


def write_band_size(uq_methods, classes, results, out_path,
                    name='uq_bands.csv'):
    '''Compute and write average band sizes to CSV file

    Parameters
    ----------
    uq_methods : Methods to perform evaluation for
    classes : List of class labels
    results : Array to store results in [UQs, IMGs, CLSs + 1]
    out_path : Folder to write results to
    name : CSV file name, optional

    '''
    csv = {'class': [*classes, 'all']}

    for uq_i, uq_method in enumerate(uq_methods):
        tmp = []
        for cls_i, _ in enumerate(classes):
            tmp.append(results[uq_i, :, cls_i].mean())
        tmp.append(results[uq_i, :, len(classes)].mean())

        csv[uq_method] = tmp

    pd.DataFrame(csv).to_csv(os.path.join(out_path, name), index=False)


def write_cls_nums(uq_methods, results, total, out_path,
                   name='uq_cls_nums.csv'):
    '''Compute and write average class set size to file

    Parameters
    ----------
    uq_methods : Methods to perform evaluation for
    results : Array to store results in [UQs]
    total : Number of images
    out_path : Folder to write results to
    name : CSV file name, optional

    '''
    csv = {}

    for uq_i, uq_method in enumerate(uq_methods):
        csv[uq_method] = [results[uq_i] / total]

    pd.DataFrame(csv).to_csv(os.path.join(out_path, name), index=False)


def assess_one_cls_perf(classes, class_set, target, uq_i, img_i, confusion,
                        drop_stats):
    '''Focus on pixels where only one class was selected and compute confusion
    matrix for those pixels, as well as pixel statistics for computing the
    class-wise recall

    Parameters
    ----------
    classes : List of class labels
    class_set : Indication of which classes were selected based on intervals
    target : Tensor of targets
    uq_i : Uncertainty method id
    img_i : Selected image id
    confusion : Confusion matrix [UQs, IMGs, 1, true class, predicted class]
    drop_stats : Pixel drop ratios [UQs, 2*CLSs + 2]

    '''
    cls_set_sizes = class_set.sum(axis=0).flatten()
    target = target.flatten()
    predicted = class_set.argmax(axis=0).flatten()

    selected_target = target[cls_set_sizes == 1]
    selected_predict = predicted[cls_set_sizes == 1]

    uqs, _, steps, true, pred = confusion.shape
    tmp_conf = np.zeros((2, uqs, steps, true, pred))
    calculate_step(tmp_conf, uq_i, classes, selected_predict, selected_target,
                   np.array([]), 0)
    confusion[uq_i, img_i, 0] = tmp_conf[0, uq_i, 0]

    for cls_i in range(len(classes)):
        cls_pred = selected_predict == cls_i
        cls_gt = selected_target == cls_i
        drop_stats[uq_i, 2*cls_i] += np.sum(cls_pred & cls_gt)
        drop_stats[uq_i, 2*cls_i+1] += np.sum(target == cls_i)

    drop_stats[uq_i, 2*len(classes)] += np.sum(cls_set_sizes == 1)
    drop_stats[uq_i, 2*len(classes)+1] += len(target)


def write_one_cls_perf(uq_methods, classes, confusion, drop_stats, out_path,
                       name='uq_drop'):
    '''Compute and write performance evaluation and drop ratios and write them
    to file

    Parameters
    ----------
    uq_methods : Methods to perform evaluation for
    classes : List of class labels
    confusion : Confusion matrix [UQs, IMGs, 1, true class, predicted class]
    drop_stats : Pixel drop stats [UQs, 2*CLSs + 2]
    out_path : Folder to write results to
    name : CSV file prefix, optional

    '''
    perf_csv = {'metric': [f'f_{cls_id}' for cls_id in classes]}
    perf_csv['metric'].append('mcc')
    drop_csv = {'class': [*classes, 'all']}

    np.savez(os.path.join(out_path, f'{name}.npz'), confusion=confusion,
             drop_stats=drop_stats)

    summary_conf = confusion.sum(axis=1)
    for uq_i, uq_method in enumerate(uq_methods):
        tmp = []
        tmp_drop = []
        for cls_i in range(len(classes)):
            tmp.append(compute_fmes(summary_conf[uq_i, 0], cls_i))
            tmp_drop.append(drop_stats[uq_i, 2*cls_i]/drop_stats[uq_i, 2*cls_i+1])
        tmp.append(compute_mcc(summary_conf[uq_i, 0]))
        tmp_drop.append(drop_stats[uq_i, 2*len(classes)]/drop_stats[uq_i, 2*len(classes)+1])
        perf_csv[uq_method] = tmp
        drop_csv[uq_method] = tmp_drop

    pd.DataFrame(perf_csv).to_csv(os.path.join(out_path, f'{name}_perf.csv'),
                                  index=False)
    pd.DataFrame(drop_csv).to_csv(os.path.join(out_path, f'{name}_drop.csv'),
                                  index=False)


def evaluate_conformal_prediction(uq_methods, selected_ids, gt_path, map_path,
                                  classes, out_path):
    '''Evaluate effectiveness and efficiency of conformal prediction
    approaches, together with thresholded performance

    Parameters
    ----------
    uq_methods : Methods to perfor evaluation for
    selected_ids : List of images to perform calculation on
    gt_path : Path to the folder with the groundtruth images
    map_path : Path to the precomputed uncertainty maps
    classes : List of class labels
    out_path : Folder to write results to

    '''
    #   - derive class sets and compute total and class-wise accuracy
    #   - compute class wise and total band length
    #   - compute average number of classes in class set
    #   - remove pixels with more than one class compute drop out ratio,
    #     f-measure and MCC
    #   - combine ditch and stream to channel

    # collect accuracy statistics
    tmp_acc = np.zeros((len(uq_methods) + 1, len(classes) + 1))

    # collect band statistics
    avg_bands = np.zeros((len(uq_methods), len(selected_ids),
                          len(classes) + 1))

    # collect class set size statistics
    tmp_cls_nums = np.zeros((len(uq_methods)))

    # collect confusion matrices for single predictions
    tmp_conf = np.zeros((len(uq_methods), len(selected_ids), 1, len(classes),
                         len(classes)))
    # collect drop ratio per class and image
    tmp_drop = np.zeros((len(uq_methods), 2*len(classes) + 2))

    for img_i, img_name in enumerate(selected_ids):
        target = tifffile.imread(os.path.join(gt_path, f'{img_name}.tif'))
        # tmp_acc[len(uq_methods), len(classes)] += np.prod(target.shape)

        for uq_i, uq_method in enumerate(uq_methods):
            data = np.load(os.path.join(map_path,
                                        f'{uq_method}_{img_name}.npz'))
            # fix different storage
            if uq_method == 'fcp':
                intervals = data['intervals'].reshape((len(classes),
                                                       *target.shape, 2))
            else:
                intervals = data['intervals']
            class_set = derive_class_set(intervals[:, :, :, 1],
                                         intervals[:, :, :, 0],
                                         classes)
            assess_correctness(tmp_acc, class_set, target, uq_methods, classes,
                               uq_i)
            assess_band_size(avg_bands, classes, uq_i, img_i, intervals)
            # compute average number of classes in class set
            tmp_cls_nums[uq_i] += class_set.sum(axis=0).mean()
            # compute performance when removing pixels with more than one class
            assess_one_cls_perf(classes, class_set, target, uq_i, img_i,
                                tmp_conf, tmp_drop)

    write_correctness(uq_methods, classes, tmp_acc, out_path)
    write_band_size(uq_methods, classes, avg_bands, out_path)
    write_cls_nums(uq_methods, tmp_cls_nums, len(selected_ids), out_path)
    write_one_cls_perf(uq_methods, classes, tmp_conf, tmp_drop, out_path)


def evaluate_uncertainty(selected_ids, classes, gt_path, map_path, out_path):
    '''Performm uncertainty quantification evaluation

    Parameters
    ----------
    selected_ids : List of images to perform calculation on
    classes : List of class labels
    gt_path : Path to the folder with the groundtruth images
    map_path : Path to the precomputed uncertainty maps
    out_path : Folder to write results to

    '''
    # sparse_steps = np.arange(0.0, 1.001, 0.001)
    sparse_steps = np.arange(0.0, 1.001, 0.01)
    # sparse_steps = np.concatenate([sparse_steps, np.arange(0.1, 1.05, 0.05)])
    results = np.zeros((2, len(UQ), len(sparse_steps), len(classes),
                        len(classes)))
    indiv_results = np.zeros((2, len(UQ), len(sparse_steps), len(classes),
                             len(classes)))
    uncertainties = -1 * np.ones((len(UQ), len(sparse_steps), 2))

    counts = []
    for uq_i, uq_method in enumerate(UQ):
        counts = calcualte_sparsification(results, indiv_results,
                                          uncertainties, uq_i, selected_ids,
                                          gt_path, map_path, classes,
                                          sparse_steps)

    # compute performance per class and for all classes
    perf = compute_perf(results[0], classes)
    # compute corrected performance per class and for all classes
    corr = compute_perf(results[1], classes)
    # compute corected streams
    stream_corr = compute_perf(indiv_results[0], classes)
    # compute corected ditches
    ditch_corr = compute_perf(indiv_results[1], classes)


    # compute AUSE
    perf_err = np.zeros((len(UQ)-1, *perf.shape[1:]))
    oracle_idx = len(UQ) - 1
    for uq_i in range(len(UQ)-1):
        perf_err[uq_i] = perf[oracle_idx] - perf[uq_i]
    # compute area under F1 score/MCC error curve per class and for all classes
    auses = compute_auses(perf_err, sparse_steps)

    # compute AUCE
    corr_err = np.zeros((len(UQ)-1, *corr.shape[1:]))
    for uq_i in range(len(UQ)-1):
        corr_err[uq_i] = corr[oracle_idx] - corr[uq_i]
    # compute area under F1 score/MCC error curve per class and for all classes
    auces = compute_auses(corr_err, sparse_steps)

    print(auses)
    print(counts)
    print(auces)
    # for plotting
    np.savez(os.path.join(out_path, 'results.npz'), perf=perf,
             uncertainties=uncertainties, auses=auses, corr=corr,
             stream_corr=stream_corr, ditch_corr=ditch_corr,
             counts=counts, auces=auces)
    # extract mean and std uncertainty for each method at each step
    # extract per class and all class IAM for each method at each step
    for uq_i, uq_method in enumerate(UQ[:-1]):
        with open(os.path.join(out_path, f'plot_{uq_method}.csv'), 'w',
                  encoding='utf-8') as out:
            for i, step in enumerate(sparse_steps):
                # skip nans
                if step == 1.0:
                    break
                # 1) step number
                line = f'{step},'
                # 2) MCC for UQ method
                line += f'{perf[uq_i, i, -1]},'
                # 3) MCC for oracle
                line += f'{perf[oracle_idx, i, -1]},'
                # 4) uncertainty mean
                uq_mean = uncertainties[uq_i, i, 0]
                line += f'{uq_mean},'
                # 5) uncertainty std
                uq_std = uncertainties[uq_i, i, 1]
                line += f'{uq_std},'
                # 6) correction: F1 background
                line += f'{corr[uq_i, i, 0]},'
                # 7) correction: F1 ditch
                line += f'{corr[uq_i, i, 1]},'
                # 8) correction: F1 stream
                line += f'{corr[uq_i, i, 2]},'
                # 9) correction: MCC
                line += f'{corr[uq_i, i, 3]},'
                # 10) correction: MCC for oracle
                line += f'{corr[oracle_idx, i, 3]},'
                # 11) stream correction: F1 ditch
                line += f'{stream_corr[uq_i, i, 1]},'
                # 12) ditch correction: F1 stream
                line += f'{ditch_corr[uq_i, i, 2]}\n'
                out.write(line)


def main(selected_ids, gt_path, map_path, out_path, uncertainty, conformal,
         classes):

    with open(selected_ids, 'r', encoding='UTF-8') as file:
        ids = [line.rstrip() for line in file]

    classes = dh.MultibandDataset.parse_classes(classes)

    if uncertainty:
        evaluate_uncertainty(ids, classes, gt_path, map_path, out_path)

    if conformal:
        # evaluate conformal prediction:
        evaluate_conformal_prediction(CONF_UQ, ids, gt_path, map_path, classes,
                                      out_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                        description='Derive the sparsification curves for all '
                        'uncertainty maps and the oracle. Additionaly, derive '
                        'the respective sparsification error curve and the '
                        'respective areas under the '
                        'sparsification error curve.',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('selected_ids', help='File containing image names for '
                        'generating uncertainty maps')
    parser.add_argument('gt_path', help='Path to groundtruth image folder')
    parser.add_argument('map_path', help='Path to folder where uncertainty '
                        'maps are stored')
    parser.add_argument('out_path', help='Path to folder where evaluation '
                        'files will be stored')
    parser.add_argument('-u', '--uncertainty', action='store_true',
                        help='Evaluate uncertainty quantification methods')
    parser.add_argument('-c', '--conformal', action='store_true',
                        help='Evaluate conformal prediction')
    parser.add_argument('--classes', help='List of class labels in ground '
                        'truth - order needs to correspond to weighting order',
                        default='0,1,2')

    args = vars(parser.parse_args())
    main(**args)
