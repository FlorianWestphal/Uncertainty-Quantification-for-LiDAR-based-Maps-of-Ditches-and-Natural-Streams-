import logging
import os

import cv2
import numpy as np
import tifffile
import torch
import torchvision

class RandomFlip:

    '''Randomly apply flip transform to inputs and target'''

    @classmethod
    def _flip_image(cls, image, flip_code):
        '''Flip given image according to given flip code

        Parameters
        ----------
        image : Image to flip
        flip_code : Indication how to flip the image

        Returns
        -------
        Flipped image

        '''
        if flip_code is None:
            return image

        return cv2.flip(image, flip_code)

    def __init__(self, rng):
        '''Setup flip transform

        Parameters
        ----------
        rng : Random number generator to use


        '''
        self._rng = rng

    def __call__(self, sample):
        '''Apply flip transform

        Parameters
        ----------
        sample : Sample dictionary of form: {'inputs': [ndarray, ...],
                 'target': ndarray}

        Returns
        -------
        Transformed sample

        '''
        flip_code = self._rng.choice([0, 1, -1, None])
        transformed_inputs = []
        for original_input in sample['inputs']:
            transformed_inputs.append(self._flip_image(original_input,
                                                       flip_code))

        transformed_target = self._flip_image(sample['target'], flip_code)

        return {'inputs': transformed_inputs, 'target': transformed_target,
                'id': sample['id']}


class RandomRotate:

    '''Apply random rotation to inputs and target images'''

    @classmethod
    def _warp_img(cls, image, transform, discretize):
        '''Warp given image based on given transform

        Parameters
        ----------
        image : Image to warp
        transform : Transformation matrix
        discretize : Indicate if resulting values should be discretized

        Returns
        -------
        Warped image

        '''
        y, x = image.shape[:2]

        border_value = 0
        warped_img = cv2.warpAffine(image, transform, dsize=(x, y),
                                    borderValue=border_value)
        if discretize:
            warped_img = np.round(warped_img)

        return warped_img

    def __init__(self, rng):
        '''Setup rotation transform

        Parameters
        ----------
        rng : Random number generator to use


        '''
        self._rng = rng

    def __call__(self, sample):
        '''Apply rotation transform

        Parameters
        ----------
        sample : Sample dictionary of form: {'inputs': [ndarray, ...],
                 'target': ndarray}

        Returns
        -------
        Transformed sample

        '''
        angle = self._rng.choice([0, 90, 180, 270])
        center = np.array(sample['target'].shape) / 2
        transform_matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1)

        transformed_inputs = []
        for original_input in sample['inputs']:
            transformed_inputs.append(self._warp_img(original_input,
                                                     transform_matrix, False))
        transformed_target = self._warp_img(sample['target'],
                                            transform_matrix, True)

        return {'inputs': transformed_inputs, 'target': transformed_target,
                'id': sample['id']}


class ToOnehot:

    def __init__(self, classes):
        '''Setup Transformation

        Parameters
        ----------
        classes : String or list of integer class labels to be found in
                  groundtruth images

        '''
        self._classes = MultibandDataset.parse_classes(classes)

    def __call__(self, sample):
        '''Apply target transformation

        Parameters
        ----------
        sample : Sample dictionary of form: {'inputs': [ndarray, ...],
                 'target': ndarray}

        Returns
        -------
        Transformed sample

        '''
        target = sample['target']
        height, width = target.shape
        onehot = np.zeros((len(self._classes), height, width))
        for idx, label in enumerate(self._classes):
            onehot[idx, target == label] = 1

        return {'inputs': sample['inputs'], 'target': onehot, 'id':
                sample['id']}


class ToOnehotGaussianBlur:

    '''Convert the label map to onehots and then use GaussianBlur to smooth the
    onehots.'''

    epsilon = 1e-6

    def __init__(self, kernel_size, classes, blur=True):
        '''Setup target conversion

        Parameters
        ----------
        kernel_size : Size of the Gaussian kernel
        classes : String or list of integer class labels to be found in
                  groundtruth images
        blur : Apply Gaussian blur, optional

        '''
        assert kernel_size // 2 != 0, "the kernel size should be odd!"
        class_number = len(MultibandDataset.parse_classes(classes))
        self._class_number = class_number
        self._blur = blur

        # adopt from cv2.getGaussianKernel
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        self._gaussian_blur = torchvision.transforms.GaussianBlur(kernel_size,
                                                                  sigma)

    def _convert_onehot(self, target):
        '''Convert target image to onehot encoding

        Parameters
        ----------
        target : Target image

        Returns
        -------
        Onehot encoding

        '''
        size = list(target.shape)
        target = target.view(-1)
        ones = torch.eye(self._class_number, device=target.device).float()
        ones = ones * (1 - self.epsilon) + (1 - ones) * self.epsilon
        onehots = (ones.index_select(0, target)
                   .view(*size, self._class_number)
                   .permute(2, 0, 1))

        return onehots

    def __call__(self, sample):
        '''Apply target transformation

        Parameters
        ----------
        sample : Sample dictionary of form: {'inputs': [ndarray, ...],
                 'target': ndarray}

        Returns
        -------
        Transformed sample

        '''
        onehots = self._convert_onehot(sample['target'])
        if self._blur:
            blurred = self._gaussian_blur(onehots)
        else:
            blurred = onehots

        # the first log makes the value doman [-inf, 0],
        # the second log makes it [-inf, inf]

        double_log_blurred = torch.log(-torch.log(blurred))

        sample['target'] = double_log_blurred

        return sample


class ToTensor:

    '''Convert inputs and target to torch tensors'''

    def __call__(self, sample):
        '''Apply transformation

        Parameters
        ----------
        sample : Sample dictionary of form: {'inputs': [ndarray, ...],
                 'target': ndarray}

        Returns
        -------
        Transformed sample

        '''
        inputs = np.zeros((len(sample['inputs']), *sample['inputs'][0].shape))
        for i, image in enumerate(sample['inputs']):
            inputs[i, :, :] = image

        if sample['target'] is np.nan:
            target = np.nan
        else:
            target = torch.from_numpy(sample['target']).type(torch.int32)
        return {'inputs': torch.from_numpy(inputs).type(torch.float),
                'target': target, 'id': sample['id']}


class MultibandDataset(torch.utils.data.Dataset):

    '''Torch dataset for loading multi-band images'''

    @classmethod
    def parse_classes(cls, classes):
        '''Convert class string to list if necessary

        Parameters
        ----------
        classes : String or list of integer classes

        Returns
        -------
        List of integer classes

        '''
        if isinstance(classes, str):
            classes = [int(f) for f in classes.split(',')]

        return classes

    @classmethod
    def _read_paths(cls, img_paths, selected, gt_path=None):
        '''Create data structure to store paths to input and groundtruth images

        Parameters
        ----------
        img_paths : List of paths to folders containing the respective band
                    images
        selected : File containing image names to be produced by dataset
        gt_path : Path to the folder with the groundtruth images, optional

        Returns
        -------
        List of image tuples ((band1_img, band2_img), gt_img, img_id)

        '''
        paths = []
        with open(selected, 'r', encoding='UTF-8') as file:
            selected_ids = [line.rstrip() for line in file]

        for selected_id in selected_ids:
            inputs = [os.path.join(img_path, f'{selected_id}.tif')
                      for img_path in img_paths]
            if gt_path is None:
                target = None
            else:
                target = os.path.join(gt_path, f'{selected_id}.tif')
            paths.append((inputs, target, selected_id))

        # check that files exist
        for inputs, target, _ in paths:
            file_paths = [*inputs, target]
            for file_path in file_paths:
                if file_path is None:
                    continue
                assert os.path.isfile(file_path), f'Missing file: {file_path}'

        return paths

    def __init__(self, img_paths, classes, selected, gt_path=None,
                 transform=None):
        '''Configure dataset

        Parameters
        ----------
        img_paths : List of paths to folders containing the respective band
                    images
        classes : String or list of integer class labels to be found in
                  groundtruth images
        selected : File containing image paths to be produced by dataset
        gt_path : Path to the folder with the groundtruth images - classes need
                  to be encoded by integers, optional
        transform : Transform to be applied on a sample, optional

        Returns
        -------
        Torch dataset

        '''
        self.paths = self._read_paths(img_paths, selected, gt_path)
        self.classes = self.parse_classes(classes)
        self.input_shape = self._infer_input_shape()
        self.transform = transform

        logging.info('Found %s images', len(self.paths))
        logging.info("Assumed image shape: %s", self.input_shape)

        torch.utils.data.Dataset.__init__(self)

    def _infer_input_shape(self):
        '''Infer input shape from ground truth image

        Returns
        -------
        input shape

        '''
        # assume all images have the same shape
        img = tifffile.imread(self.paths[0][0][0])

        return (len(self.paths[0][0]), *img.shape)

    def __len__(self):
        '''Return number of images in the dataset
        Returns
        -------
        Number of images

        '''
        return len(self.paths)

    def __getitem__(self, idx):
        '''Return input and groundtruth image corresponding to the given index

        Parameters
        ----------
        idx : Index of the image to retrieve

        Returns
        -------
        Requested sample dict {'inputs': ndarray, 'target': ndarray}

        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_paths, gt_path, img_id = self.paths[idx]

        inputs = []
        for input_path in input_paths:
            inputs.append(tifffile.imread(input_path))
        if gt_path is None:
            target = np.nan
        else:
            target = tifffile.imread(gt_path)

        sample = {'inputs': inputs, 'target': target, 'id': img_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def infer_weights(self, weighting):
        '''Infer weigthing based on dataset and given weighting mode

        Parameters
        ----------
        weighting : Weighting mode [none|mfb|weight string]

        Returns
        -------
        Numpy array with one weight per class or None

        '''
        if weighting == 'none':
            weights = None
        elif weighting == 'mfb':
            weights = self._infer_mfb_weights()
        else:
            weights = np.array([float(f) for f in weighting.split(',')])

        logging.info('Set weights to: %s', weights)
        return weights

    def _infer_mfb_weights(self):
        '''Calculate weights using median frequency balancing.
           This approach has been proposed by Eigen and Fergus.
           When there is an uneven number of classes, the weight for the class
           with the median frequency is set to 1.0.

           D. Eigen, and R. Fergus, "Predicting Depth, Surface Normals and
           Semantic Labels With a Common Multi-Scale Convolutional
           Architecture", 2015,

        Returns
        -------
        list of class weights based on median frequency balancing

        '''
        weights = []
        class_counts = {c: 0 for c in self.classes}
        class_totals = {c: 0 for c in self.classes}

        for _, target_path, _ in self.paths:
            target = tifffile.imread(target_path)
            target_pixels = np.prod(target.shape)
            for cls_idx in self.classes:
                class_counts[cls_idx] += np.sum(target == cls_idx)
                # only count the image if it contains at least one instance of
                # the current class
                if (target == cls_idx).any():
                    class_totals[cls_idx] += target_pixels

        frequencies = []
        for cls_idx in self.classes:
            cnt = class_counts[cls_idx]
            tot = class_totals[cls_idx]
            if cnt == 0 and tot == 0:
                logging.warning('WARN: No pixel of class %s found.', cls_idx)
                # This case should not happen. However, if it does, setting the
                # frequency to a small value ensures that the code can run.
                frequencies.append(0.0001)
            elif cnt > tot:
                raise ValueError(f'Count mismatch class: {cls_idx} '
                                 f'Count: {cnt} Total: {tot}')
            else:
                frequencies.append(cnt/tot)

        weights = [np.median(frequencies) / freq for freq in frequencies]

        return np.array(weights)
