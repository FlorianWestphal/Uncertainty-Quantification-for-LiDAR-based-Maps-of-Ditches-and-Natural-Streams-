import math
import os

import concretedropout.pytorch as cd
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


class IoU:

    '''Class-wise intersecion over union metric'''

    def __init__(self, classes, regression=False):
        '''Setup IoU metric

        Parameters
        ----------
        classes : List of classes (int)
        regression : Compute metric for regression problem
        '''
        self._regression = regression
        self._results = {f: [] for f in classes}

    def reset(self):
        '''Reset internal datastructure
        '''
        for key in self._results:
            self._results[key] = []

    def add(self, predicted, target):
        '''Add result for predicted and target pair to metric

        Parameters
        ----------
        predicted : Tensor of predicted segmentation
        target : Tensor of targets

        '''
        # undo log transform (may not be necessary)
        if self._regression:
            predicted = torch.exp(-torch.exp(predicted))
            target = torch.exp(-torch.exp(target))

        predicted = (predicted.argmax(dim=1)
                     .view(-1))
        target = (target.argmax(dim=1)
                  .view(-1))

        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        for label in self._results:
            pred_label = predicted == label
            target_label = target == label

            intersect = np.logical_and(pred_label, target_label).sum()
            union = np.logical_or(pred_label, target_label).sum()

            if union > 0:
                self._results[label].append(intersect/union)

    def value(self):
        '''Compute class-wise IoU
        Returns
        -------
        Class - IoU map

        '''
        results = {}

        for label in self._results:
            results[label] = np.mean(self._results[label])

        return results


class WeightedMSE(nn.Module):

    '''Weighted mean square error loss'''

    def __init__(self, weights=None, device='cuda'):
        '''Setup loss

        Parameters
        ----------
        weights : List of class weights, optional
        device : Select device to run on, optional
        '''
        nn.Module.__init__(self)
        self._weights = torch.from_numpy(weights).to(device)

    def forward(self, inputs, targets):
        '''Compute loss for given inputs and targets

        Parameters
        ----------
        inputs : Predicted output
        targets : Target output

        Returns
        -------
        Weighted MSE

        '''
        if self._weights is not None:
            mse = ((inputs - targets) ** 2).mean(dim=(0, 2, 3))
            mse = (self._weights * mse).sum()
        else:
            mse = ((inputs - targets) ** 2).mean()

        return mse


class ModelCheckpoint:

    '''Save best performing model'''

    def __init__(self, log_path, model, optimizer,
                 file_name='minloss_checkpoint.pth.tar'):
        '''Setup checkpoiting

        Parameters
        ----------
        log_path : Path to store model file to
        model : Model reference
        optimizer : Optimizer reference
        file_name : Name of the checkpoint

        '''
        self._model_path = os.path.join(log_path, file_name)
        self._model = model
        self._optimizer = optimizer
        self._minloss = np.infty

    def update(self, epoch, loss):
        '''Save model if given loss improved

        Parameters
        ----------
        epoch : Current epoch number
        loss : New loss value

        '''
        if loss < self._minloss:
            self._minloss = loss
            state = {'epoch': epoch, 'state_dict': self._model.state_dict(),
                     'opt_dict': self._optimizer.state_dict()}
            torch.save(state, self._model_path)


class CSVLogger:

    '''Write training log'''

    def __init__(self, log_path, metric, file_name='log.csv'):
        '''Setup logging

        Parameters
        ----------
        log_path : Path to store CSV log
        metric : Metric to log
        file_name : Name of the log file

        '''
        self._file = os.path.join(log_path, file_name)
        self._metric_order = [f for f in metric.value()]

        columns = ['epoch', 'loss']
        columns += [f'iou_{f}' for f in self._metric_order]
        columns += ['lr', 'val_loss']
        columns += [f'val_iou_{f}' for f in self._metric_order]

        with open(self._file, 'w', encoding='utf-8') as log:
            log.write(','.join(columns) + '\n')

    def update(self, epoch, train_loss, train_metric, valid_loss, valid_metric,
               learning_rate):
        '''Write current progress

        Parameters
        ----------
        epoch : Current epoch number
        train_loss : Training loss
        train_metric : Training metric results
        valid_loss : Validation loss
        valid_metric : Validation metric results
        learning_rate : Current learning rate

        '''
        entry = [epoch, train_loss]
        entry += [train_metric[f] for f in self._metric_order]
        entry += [learning_rate, valid_loss]
        entry += [valid_metric[f] for f in self._metric_order]

        with open(self._file, 'a', encoding='utf-8') as log:
            log_entry = ','.join(str(el) for el in entry)
            log.write(log_entry + '\n')


class BaseUNet(nn.Module):

    '''UNet architecture class'''

    LAYER_CONFIG = [32, 64, 128, 256]

    @classmethod
    def _pad(cls, inputs):
        '''Add padding if needed

        Parameters
        ----------
        inputs : Input tensor

        Returns
        -------
        Padded tensor, applied padding

        '''
        _, _, height, width = inputs.size()

        width_correct = 2**math.ceil(math.log2(width)) - width
        height_correct = 2**math.ceil(math.log2(height)) - height

        left = width_correct // 2
        right = width_correct - left
        top = height_correct // 2
        bottom = height_correct - top

        padding = (top, bottom, left, right)
        return nn.functional.pad(inputs, (left, right, top, bottom)), padding

    @classmethod
    def _unpad(cls, inputs, padding):
        '''TODO: Docstring for _unpad.

        Parameters
        ----------
        inputs : Input tensor
        padding : Padding information

        Returns
        -------
        Tensor without padding

        '''
        top, bottom, left, right = padding

        bottom = -1 if bottom == 0 else -bottom
        right = -1 if right == 0 else -right

        return inputs[:, :, top:bottom, left:right]

    @classmethod
    def _downsample(cls, inputs, layers, act, pool, steps_per_layer=3):
        '''Downsample inputs

        Parameters
        ----------
        inputs : Input tensor
        layers : Layer steps
        act : Activation function to use
        pool : Pooling function to use
        steps_per_layer : Computation steps per layer, optional

        Returns
        -------
        Output tensor, skip tensors

        '''
        result = inputs
        skips = []

        layer_num = len(layers) // steps_per_layer
        for layer_i in range(layer_num):
            conv1 = layers[layer_i*steps_per_layer]
            drop = layers[layer_i*steps_per_layer+1]
            conv2 = layers[layer_i*steps_per_layer+2]

            result = drop(result, conv1)
            result = act(result)
            result = conv2(result)
            result = act(result)
            skips.append(result)
            result = pool(result)

        return result, skips

    @classmethod
    def _process(cls, inputs, layers, act):
        '''Implement lowest level in UNet

        Parameters
        ----------
        inputs : Input tensor
        layers : Layer steps
        act : Activation function to use

        Returns
        -------
        Output tensor

        '''
        conv1 = layers[0]
        drop = layers[1]
        conv2 = layers[2]

        result = drop(inputs, conv1)
        result = act(result)
        result = conv2(result)
        result = act(result)

        return result

    @classmethod
    def _upsample(cls, inputs, skips, layers, act, steps_per_layer=4):
        '''Upsample inputs

        Parameters
        ----------
        inputs : Input tensor
        skips : Skip tensors
        layers : Layer steps
        act : Activation function to use
        steps_per_layer : Computation steps per layer, optional

        Returns
        -------
        Output tensor

        '''
        # auto_LiRPA seems to sometimes run inputs on CPU and sometimes on GPU,
        # which collides with the skip tensors
        if not inputs.is_cuda:
            tmp = []
            for skip in skips:
                tmp.append(skip.cpu())
            skips = tmp
        result = inputs

        layer_num = len(layers) // steps_per_layer
        for layer_i in range(layer_num):
            tconv = layers[layer_i*steps_per_layer]
            conv1 = layers[layer_i*steps_per_layer+1]
            drop = layers[layer_i*steps_per_layer+2]
            conv2 = layers[layer_i*steps_per_layer+3]

            result = tconv(result)
            # result = torch.cat((result, skips[-1 * (layer_i+1)]), dim=1)
            result = torch.cat((result, skips.pop()), dim=1)
            result = drop(result, conv1)
            result = act(result)
            result = conv2(result)
            result = act(result)

        return result


class UNet(BaseUNet):

    '''UNet implementation for uncertainty quantification using feature
    conformal prediction and concrete dropout'''

    def __init__(self, in_channels, classes, device='cuda'):
        '''Setup model

        Parameters
        ----------
        in_channels : Number of input channels
        classes : List of class labels
        device : Device to run model on [cuda|cpu]
        '''
        BaseUNet.__init__(self)

        # ensure that all tensors are created on the same device
        # torch.set_default_device(device)
        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self._classes = classes
        self._device = device
        self._downsampling = nn.ModuleList()
        self._processing = nn.ModuleList()
        self._upsampling = nn.ModuleList()
        self._act = nn.ReLU()
        self._pool = nn.MaxPool2d(2)

        # downsampling
        for filters in self.LAYER_CONFIG:
            self._downsampling.append(nn.Conv2d(in_channels, filters, 3,
                                      padding='same'))
            in_channels = filters
            self._downsampling.append(cd.ConcreteDropout2D())
            self._downsampling.append(nn.Conv2d(in_channels, filters, 3,
                                      padding='same'))

        # processing
        self._processing.append(nn.Conv2d(in_channels, 512, 3, padding='same'))
        in_channels = 512
        self._processing.append(cd.ConcreteDropout2D())
        self._processing.append(nn.Conv2d(in_channels, 512, 3, padding='same'))

        # upsampling
        for filters in reversed(self.LAYER_CONFIG):
            self._upsampling.append(nn.ConvTranspose2d(in_channels, filters, 2,
                                    stride=2))
            in_channels = filters
            self._upsampling.append(nn.Conv2d(2*in_channels, filters, 3,
                                    padding=1))
            self._upsampling.append(cd.ConcreteDropout2D())
            self._upsampling.append(nn.Conv2d(in_channels, filters, 3,
                                    padding=1))

        # output
        self._output = nn.Conv2d(in_channels, len(self._classes), 1)

        # initialize conv layers
        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, layer):
        '''Initialize weights

        Parameters
        ----------
        layer : Layer to initialize

        '''
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.ConvTranspose2d):
            nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        '''Run model on given input

        Parameters
        ----------
        inputs : Input tensor

        Returns
        -------
        Predicted segmentation

        '''
        result, padding = self._pad(inputs)

        # downsampling
        result, skips = self._downsample(result, self._downsampling,
                                         self._act, self._pool)

        # processing
        result = self._process(result, self._processing, self._act)

        # upsampling
        result = self._upsample(result, skips, self._upsampling, self._act)

        result = self._output(result)

        return self._unpad(result, padding)

    def load(self, model_path):
        '''Load model weights from given path

        Parameters
        ----------
        model_path : Path to the model

        '''
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['state_dict'])

    def fit(self, train_it, valid_it, epochs, log_dir, weights):
        '''Train model

        Parameters
        ----------
        train_it : Training data
        valid_it : Validation data
        epochs : Number of epochs to train
        log_dir : Folder to which logging information is saved
        weights : List of class weights

        '''

        criterion = WeightedMSE(weights)
        # criterion = torch.nn.CrossEntropyLoss(torch.from_numpy(
        #                                             np.array(weights))
        #                                       .to(self._device))
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        metric = IoU(self._classes, True)
        checkpoint = ModelCheckpoint(log_dir, self, optimizer)
        csv_log = CSVLogger(log_dir, metric)

        for epoch in tqdm(range(epochs)):

            train_loss, train_iou = self._train(train_it, criterion,
                                                optimizer, metric)
            scheduler.step(train_loss)

            valid_loss, valid_iou = self._validate(valid_it, criterion, metric)
            checkpoint.update(epoch, valid_loss)
            csv_log.update(epoch, train_loss, train_iou, valid_loss, valid_iou,
                           optimizer.param_groups[0]['lr'])

    def proba(self, batch):
        '''Predict class probabilities for given batch

        Parameters
        ----------
        batch : Input images

        Returns
        -------
        Probability map

        '''
        self.eval()
        if isinstance(batch, dict):
            batch = batch['inputs'].to(self._device,
                                       dtype=batch['inputs'].dtype)
        predicted = self(batch)
        predicted = torch.exp(-torch.exp(predicted))

        # softmax = torch.nn.Softmax2d()
        # predicted = softmax(predicted)

        return predicted.detach().cpu().numpy()

    def predict(self, batch):
        '''Predict classes for given batch

        Parameters
        ----------
        batch : Input images

        Returns
        -------
        Class map

        '''
        predicted = self.proba(batch)

        return predicted.argmax(axis=1)

    def _set_mc_dropout(self, enable):
        '''Configure model to enable MC Dropout

        Parameters
        ----------
        enable : Set to True or False

        '''
        for module in filter(lambda x: isinstance(x, cd.ConcreteDropout2D),
                             self.modules()):
            module.is_mc_dropout = enable

    def mc_dropout(self, batch, samples, batch_size):
        '''Perform Monte Carlo Dropout to infer map of predictive entropy and
        mutual information

        Parameters
        ----------
        batch : Input images
        samples : Number of Monte Carlo samples
        batch_size : Maximum batch size to process at once

        Returns
        -------
        Predictive entropy map, Mutual information map

        '''
        self._set_mc_dropout(True)
        image = batch['inputs'].to(self._device, dtype=batch['inputs'].dtype)
        assert image.shape[0] == 1, 'Process one image at a time'

        batch = image.repeat(batch_size, 1, 1, 1)

        repeats = samples // batch_size
        outputs = []
        for _ in range(repeats):
            # predict
            predicted = self.proba(batch)
            outputs.append(predicted)

        pred_samples = np.concatenate(outputs)
        pred_distr = pred_samples.mean(axis=0)
        sample_size = pred_samples.shape[0]

        # calculate predictive entropy
        eps = np.finfo(pred_distr.dtype).tiny
        log_distr = np.log(pred_distr + eps)
        pred_entropy = -1 * np.sum(pred_distr * log_distr, axis=0)

        # calucalte mutual information
        log_samples = np.log(pred_samples + eps)
        minus_e = np.sum(pred_samples * log_samples, axis=(0, 1))
        minus_e /= sample_size
        mutual_info = pred_entropy + minus_e

        self._set_mc_dropout(False)

        return pred_entropy, mutual_info

    def _train(self, train_it, criterion, optimizer, metric):
        '''Train model on epoch

        Parameters
        ----------
        train_it : Training data
        criterion : Training criterion
        optimizer : Optimizer
        metric : Metric

        Returns
        -------
        Training loss, metric output

        '''
        self.train()
        epoch_loss = 0.0
        metric.reset()

        for batch in train_it:
            inputs = batch['inputs'].to(self._device,
                                        dtype=batch['inputs'].dtype)
            target = batch['target'].to(self._device,
                                        dtype=batch['target'].dtype)

            outputs = self(inputs)

            # add dropout regularization
            reg = torch.zeros(1)  # get the regularization term
            for module in filter(lambda x: isinstance(x, cd.ConcreteDropout2D),
                                 self.modules()):
                reg += module.regularization
            loss = criterion(outputs, target) + reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            metric.add(outputs.detach(), target.detach())

        return epoch_loss/len(train_it), metric.value()

    def _validate(self, valid_it, criterion, metric):
        '''Validate model

        Parameters
        ----------
        valid_it : Validation data
        criterion : Training criterion
        metric : Metric

        Returns
        -------
        Validation loss, validation metric

        '''
        self.eval()
        epoch_loss = 0.0
        metric.reset()

        for batch in valid_it:
            inputs = batch['inputs'].to(self._device,
                                        dtype=batch['inputs'].dtype)
            target = batch['target'].to(self._device,
                                        dtype=batch['target'].dtype)

            with torch.no_grad():
                outputs = self(inputs)
                loss = criterion(outputs, target)

            epoch_loss += loss.item()
            metric.add(outputs.detach(), target.detach())

        return epoch_loss/len(valid_it), metric.value()


class ConformalBase(BaseUNet):

    '''Encoder network for conformal model'''

    def __init__(self, downsample, process, upsample, act, pool, device):
        '''Setup encoder

        Parameters
        ----------
        downsample : Module list for downsampling
        process : Module list for processing
        upsample : Module list for upsampling
        act : Activation function to use
        pool : Pooling function to use
        '''
        nn.Module.__init__(self)

        self._downsampling = downsample
        self._processing = process
        self._upsampling = upsample
        self._act = act
        self._pool = pool

        self.to(device)

    def forward(self, inputs):
        '''Run model on given input

        Parameters
        ----------
        inputs : Input tensor

        Returns
        -------
        Extracted features

        '''
        result, padding = self._pad(inputs)

        # downsampling
        result, skips = self._downsample(result, self._downsampling,
                                         self._act, self._pool)

        # processing
        result = self._process(result, self._processing, self._act)

        # upsampling
        if len(self._upsampling) > 0:
            result = self._upsample(result, skips, self._upsampling,
                                    self._act)

        return result, skips, padding


class ConformalHead(BaseUNet):

    '''Decoder network for conformal model '''

    def __init__(self, upsampling, act, output, device):
        '''Setup decoder

        Parameters
        ----------
        upsampling : Module list for upsampling
        act : Activation function to use
        output : Output layer
        '''
        nn.Module.__init__(self)

        self._upsampling = nn.ModuleList()
        for module in upsampling:
            # self._upsampling.append(deepcopy(module))
            self._upsampling.append(module)
        self._act = act
        self._output = output
        # self._act = deepcopy(act)
        # self._output = deepcopy(output)

        self.__skips = None
        self.__cal_idx = None
        self.__val_idx = None
        self.padding = None

        self.to(device)

    def forward(self, inputs):
        '''Run model on given input

        Parameters
        ----------
        inputs : Feature representation, intermediate representations

        Returns
        -------
        Predicted segmentation

        '''
        result = inputs

        # upsampling
        # select instances to be used for upsampling
        if (self.__cal_idx is None
                or len(self.__skips) == 0
                or result.shape[0] == self.__skips[0].shape[0]):
            skips = self.__skips
        else:
            skips = []
            if result.shape[0] == len(self.__cal_idx):
                idx = self.__cal_idx
            elif result.shape[0] == len(self.__val_idx):
                idx = self.__val_idx
            else:
                raise RuntimeError(f'Unknown number of '
                                   f'instances: {result.shape[0]}')
            for skip in self.__skips:
                skips.append(skip[idx])

        result = self._upsample(result, skips, self._upsampling,
                                self._act)

        result = self._output(result)

        return result

    def update_skips(self, skips):
        '''Update tensors for skip connections

        Parameters
        ----------
        skips : Skip tensors
        to_cpu : Move skip tensors to CPU (Needed for auto_LiRPA)

        Returns
        -------
        self

        '''
        self.__skips = []
        # only copy the skips which are relevant for the upsampling included
        # in the head
        # Note: there are 4 modules for each layer in the upsampling path
        for skip in skips[:len(self._upsampling)//4]:
            skip = skip.detach()
            self.__skips.append(skip)
        return self

    def update_idxs(self, cal_idx, val_idx):
        '''Update indices of samples used for calibration and validation

        Parameters
        ----------
        cal_idx : Indices of the calibration instances
        val_idx : Indices of the validation instances

        Returns
        -------
        self

        '''
        self.__cal_idx = cal_idx
        self.__val_idx = val_idx

        return self

    def update_padding(self, padding):
        '''Update padding configuration

        Parameters
        ----------
        padding : Padding information

        Returns
        -------
        self

        '''
        self.padding = padding
        return self


class ConformalUNet(nn.Module):

    '''Model container class for feature conformal prediction based uncertainty
    quantification'''

    CALIBRATION_FILE = 'calibration_scores_dli{}.npz'

    @classmethod
    def default_loss(cls, inputs, targets):
        return torch.norm(inputs - targets) ** 2

    def __init__(self, trained_model, decoder_layer_idx,
                 upsample_steps_per_layer=4, device='cuda',
                 cert_optimizer='sgd', inv_lr=1e-2):
        '''Setup model based on pre-trained model

        Parameters
        ----------
        trained_model : Pre-trained model
        decoder_layer_idx : Index into decoder path at which to split the
                            model for feature extraction [0-4]
        upsample_steps_per_layer : Computation steps per layer, optional
        device : Device to run model on [cuda|cpu]
        cert_optimizer : Optimizer to use, optional
        inv_lr : Learning rate to use, optional
        '''
        nn.Module.__init__(self)

        layers = list(trained_model.children())
        idx = decoder_layer_idx * upsample_steps_per_layer
        self._encoder = ConformalBase(layers[0], layers[1], layers[2][:idx],
                                      layers[3], layers[4], device)
        self._decoder = ConformalHead(layers[2][idx:], layers[3], layers[5],
                                      device)

        self._feat_norm = np.inf
        self._err_func = FeatErrorErrFunc(feat_norm=np.inf)
        self._cert_optimizer = cert_optimizer
        self._inv_lr = inv_lr
        self._device = device
        self._inv_step = None
        self._decoder_layer_idx = decoder_layer_idx

        self.cal_scores = None

    def calibration_name(self):
        '''Get name of the calibration file
        Returns
        -------
        calibration file name

        '''
        return self.CALIBRATION_FILE.format(self._decoder_layer_idx)

    def log_name(self):
        '''Get log name
        Returns
        -------
        Log name

        '''
        return 'fcp'

    def load_calibration_scores(self, calibration_file):
        '''Load Calibration scores from file

        Parameters
        ----------
        calibration_file : Path to calibration file

        '''
        self.cal_scores = np.load(calibration_file)['cal_scores']

    def save_calibration_scores(self, calibration_file):
        '''Save calibration scores to file

        Parameters
        ----------
        calibration_file : Path to calibration file

        '''
        np.savez_compressed(calibration_file, cal_scores=self.cal_scores)

    def forward(self, inputs):
        '''Run model on given input

        Parameters
        ----------
        inputs : Input tensor

        Returns
        -------
        Predicted segmentation

        '''
        result, skips, padding = self._encoder(inputs)
        self._decoder.update_skips(skips).update_padding(padding)
        result = self._decoder(result)
        result = BaseUNet._unpad(result, padding)

        return result

    def calibrate(self, calib_it):
        '''Calibrate network

        Parameters
        ----------
        calib_it : Calibration data

        '''
        cal_scores = self._score(calib_it)
        self.cal_scores = np.sort(cal_scores, 0)[::-1]

    def predict(self, inputs, significance=0.1):
        '''Generate prediction intervals for given batch and set significance
           level

        Parameters
        ----------
        inputs : Input tensor
        significance : Significance level, optional

        Returns
        -------
        Prediction intervals for the batch
        [batch, classes*width*height, lower/upper bound]

        '''
        self.eval()

        n_test = inputs.shape[0]
        inputs = inputs.to(self._device).requires_grad_(False)
        # TODO this seems overkill
        predicted = self(inputs).cpu().detach().numpy()

        intervals = np.zeros((n_test, np.prod(predicted.shape[1:]), 2))
        feat_err_dist = self._err_func.apply_inverse(self.cal_scores,
                                                     significance)

        z, skips, padding = self._encoder(inputs)
        z = z.detach()
        self._decoder.update_skips(skips).update_padding(padding)

        lirpa_model = BoundedModule(self._decoder, torch.empty_like(z).cuda())
        ptb = PerturbationLpNorm(norm=np.inf, eps=feat_err_dist[0][0])
        my_input = BoundedTensor(z, ptb)

        lb, ub = lirpa_model.compute_bounds(x=(my_input,), method='IBP')
        lb = torch.exp(-torch.exp(lb))
        ub = torch.exp(-torch.exp(ub))
        lb, ub = lb.detach().cpu().numpy(), ub.detach().cpu().numpy()

        # TODO refactor
        lb = BaseUNet._unpad(lb, padding)
        ub = BaseUNet._unpad(ub, padding)
        # reverse the order, since reverting the double log changes what is the
        # upper and what is the lower bound
        intervals[..., 0] = ub.reshape((ub.shape[0], -1))
        intervals[..., 1] = lb.reshape((lb.shape[0], -1))

        return intervals

    def _score(self, calib_it):
        '''Score samples in the calibration set

        Parameters
        ----------
        calib_it : Calibration data

        Returns
        -------
        Scored samples

        '''
        if self._inv_step is None:
            self._inv_step = self._find_best_step_num(calib_it)

        print('calculating score:')
        ret_val = []
        for batch in tqdm(calib_it):
            inputs = batch['inputs'].to(self._device,
                                        dtype=batch['inputs'].dtype)
            target = batch['target'].to(self._device,
                                        dtype=batch['target'].dtype)

            norm = np.ones(len(inputs))

            z_pred, skips, padding = self._encoder(inputs)
            self._decoder.update_skips(skips).update_padding(padding)

            z_true = self._inv_g(z_pred, target, step=self._inv_step)
            z_pred = z_pred.view(z_pred.shape[0], -1)
            z_true = z_true.view(z_true.shape[0], -1)
            batch_ret_val = self._err_func.apply(z_pred.detach().cpu(),
                                                 z_true.detach().cpu())
            batch_ret_val = batch_ret_val.detach().cpu().numpy() / norm
            ret_val.append(batch_ret_val)
        ret_val = np.concatenate(ret_val, axis=0)

        return ret_val

    def _find_best_step_num(self, calib_it):
        '''Estimate number of steps needed to find "true" feature
        representation given a feature representation produced by the encoder
        and the actual labels

        Parameters
        ----------
        calib_it : Calibration data

        Returns
        -------
        Step number

        '''
        max_inv_steps = 200
        val_sig = 0.1

        acc_val_coverage = np.zeros(max_inv_steps)
        acc_val_num = 0
        print("begin to find the best step number")
        for batch in tqdm(calib_it):
            inputs = batch['inputs'].to(self._device,
                                        dtype=batch['inputs'].dtype)
            target = batch['target'].to(self._device,
                                        dtype=batch['target'].dtype)

            z_pred, skips, padding = self._encoder(inputs)
            self._decoder.update_skips(skips).update_padding(padding)
            each_step_val_coverage, val_num = self._coverage_tight(
                                                        inputs, target, z_pred,
                                                        steps=max_inv_steps,
                                                        val_sig=val_sig)
            acc_val_coverage += np.array(each_step_val_coverage) * val_num
            acc_val_num += val_num
            # release memory to avoid OOM
            del z_pred

        each_step_val_coverage = acc_val_coverage / acc_val_num

        tolerance = 3
        count = 0
        final_coverage, best_step = None, None
        for i, val_coverage in enumerate(each_step_val_coverage):
            if val_coverage > (1 - val_sig) * 100 and final_coverage is None:
                count += 1
                if count == tolerance:
                    final_coverage = val_coverage
                    best_step = i
            elif val_coverage <= (1 - val_sig) * 100 and count > 0:
                count = 0

        if final_coverage is None or best_step is None:
            raise ValueError('Cannot find a good step to make the coverage '
                             'higher than {}'.format(1 - val_sig))

        print(f'The best inv_step is {best_step+1}, which gets '
              f'{final_coverage} coverage on val set')

        return best_step + 1

    def _coverage_tight(self, inputs, target, z_pred, steps, val_sig):
        '''Estimate coverage for different numbers of optimization steps

        Parameters
        ----------
        inputs : Input tensor
        target : Tensor of targets
        z_pred : Feature representation for batch
        steps : Maximum number of steps to try
        val_sig : Validation significance

        Returns
        -------
        Coverage values for each optimizer step, Number of validation samples
        used

        '''
        z_pred_detach = z_pred.detach().clone()

        idx = torch.randperm(len(z_pred_detach))
        n_val = int(np.floor(len(z_pred_detach) / 5))
        val_idx, cal_idx = idx[:n_val], idx[n_val:]

        cal_x, val_x = inputs[cal_idx], inputs[val_idx]
        cal_y, val_y = target[cal_idx], target[val_idx]
        cal_z_pred, val_z_pred = z_pred_detach[cal_idx], z_pred_detach[val_idx]
        self._decoder.update_idxs(cal_idx, val_idx)

        cal_score_list = self._get_each_step_err_dist(cal_x, cal_y, cal_z_pred,
                                                      steps=steps)
        print('cal score')
        print(torch.cuda.memory_summary())
        val_score_list = self._get_each_step_err_dist(val_x, val_y, val_z_pred,
                                                      steps=steps)
        print('val score')
        print(torch.cuda.memory_summary())

        val_coverage_list = []
        for i, (cal_score, val_score) in enumerate(zip(cal_score_list,
                                                       val_score_list)):
            err_dist_threshold = self._err_func.apply_inverse(
                                                    nc=cal_score,
                                                    significance=val_sig)[0][0]
            val_coverage = np.sum(val_score < err_dist_threshold) * 100
            val_coverage /= len(val_score)
            val_coverage_list.append(val_coverage)

        return val_coverage_list, len(val_x)

    def _get_each_step_err_dist(self, inputs, target, z_pred, steps):
        '''Calculate the distance between the predicted feature representation
        and the "optimal" feature representation

        Parameters
        ----------
        inputs : Input tensor
        target : Tensor of targets
        z_pred : Feature representation for batch
        steps : Maximum number of steps to try

        Returns
        -------
        List of distances per step

        '''
        each_step_z_true = self._inv_g(z_pred, target, step=steps,
                                       record_each_step=True)

        norm = np.ones(len(inputs))

        err_dist_list = []
        z_pred = z_pred.view(z_pred.shape[0], -1)
        for i, step_z_true in enumerate(each_step_z_true):
            step_z_true = step_z_true.view(step_z_true.shape[0], -1)
            err_dist = self._err_func.apply(z_pred.detach().cpu(),
                                            step_z_true.detach().cpu())
            err_dist = err_dist.numpy() / norm
            err_dist_list.append(err_dist)

        return err_dist_list

    def _inv_g(self, z_pred, target, step=None, record_each_step=False):
        '''Perform given number of optimizer steps to find "optimal" feature
           representation for given samples

        Parameters
        ----------
        z_pred : Feature representation for batch
        target : Tensor of targets
        step : Number of optimizer steps to perform, optional
        record_each_step : Return feature representation for each step,
                           optional

        Returns
        -------
        "optimal" feature representation or list of "optimal" feature
        representations

        '''
        z = z_pred.detach().clone()
        z = z.detach()
        z.requires_grad_()
        if self._cert_optimizer == "sgd":
            optimizer = torch.optim.SGD([z], lr=self._inv_lr)
        elif self._cert_optimizer == "adam":
            optimizer = torch.optim.Adam([z], lr=self._inv_lr)

        self.eval()
        each_step_z = []
        for _ in range(step):
            pred = self._decoder(z)

            pred = BaseUNet._unpad(pred, self._decoder.padding)
            loss = ConformalUNet.default_loss(pred.squeeze(), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if record_each_step:
                each_step_z.append(z.detach().cpu().clone())

        if record_each_step:
            return each_step_z
        else:
            return z.detach().cpu()


class FeatErrorErrFunc:

    def __init__(self, feat_norm):
        super(FeatErrorErrFunc, self).__init__()
        self.feat_norm = feat_norm

    def apply(self, prediction, z):
        ret = (prediction - z).norm(p=self.feat_norm, dim=1)
        return ret

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1))) - 1
        border = min(max(border, 0), nc.size - 1)

        return np.vstack([nc[border], nc[border]])


class ConformalRegressor:

    '''Wrapper class for implementation of conformal regression.
       This method performs uncertaity quantification using normalized
       inductive conformal regression separately for each class.
    '''

    def __init__(self, trained_model, samples, batch_size, device='cuda'):
        '''Setup wrapper

        Parameters
        ----------
        trained_model : Pre-trained model
        samples : Number of Monte Carlo samples
        batch_size : Maximum batch size to process at once
        '''

        self._model = trained_model
        self._samples = samples
        self._batch_size = batch_size

        self.cal_scores_uncond = None
        self.cal_scores_cond = None
        self._device = device

    def calibration_name(self):
        '''Get name of the calibration file
        Returns
        -------
        calibration file name

        '''
        return 'cr_calibration_scores.npz'

    def log_name(self):
        '''Get log name
        Returns
        -------
        Log name

        '''
        return 'cr'

    def load_calibration_scores(self, calibration_file):
        '''Load Calibration scores from file

        Parameters
        ----------
        calibration_file : Path to calibration file

        '''
        data = np.load(calibration_file)
        self.cal_scores_uncond = data['cal_scores_uncond']
        self.cal_scores_cond = [[data['p_cls_0'], data['n_cls_0']], 
                                [data['p_cls_1'], data['n_cls_1']], 
                                [data['p_cls_2'], data['n_cls_2']]] 

    def save_calibration_scores(self, calibration_file):
        '''Save calibration scores to file

        Parameters
        ----------
        calibration_file : Path to calibration file

        '''
        np.savez_compressed(calibration_file,
                            cal_scores_uncond=self.cal_scores_uncond,
                            p_cls_0=self.cal_scores_cond[0][0],
                            p_cls_1=self.cal_scores_cond[1][0],
                            p_cls_2=self.cal_scores_cond[2][0],
                            n_cls_0=self.cal_scores_cond[0][1],
                            n_cls_1=self.cal_scores_cond[1][1],
                            n_cls_2=self.cal_scores_cond[2][1])

    def calibrate(self, calib_it):
        '''Calibrate network

        Parameters
        ----------
        calib_it : Calibration data

        '''
        cal_scores_uncond, cal_scores_cond = self._score(calib_it)
        for uncond, cond in zip(cal_scores_uncond, cal_scores_cond):
            uncond.sort()
            cond[0].sort()
            cond[1].sort()
        self.cal_scores_uncond = cal_scores_uncond
        self.cal_scores_cond = cal_scores_cond

    def _score(self, calib_it):
        '''Score samples in the calibration set

        Parameters
        ----------
        calib_it : Calibration data

        Returns
        -------
        Unconditionally scored samples [classes, scores], Conditionally scored
        samples [classes, scores]

        '''
        print('calculating score:')
        uncond_val = []
        cond_val = []
        self._model._set_mc_dropout(True)
        for batch in tqdm(calib_it):
            inputs = batch['inputs'].to(self._device,
                                        dtype=batch['inputs'].dtype)
            target = batch['target']

            for x, y in zip(inputs, target):
                # perform MC dropout
                means, stds = self._perform_mc_dropout(x[None, :, :, :])

                # calculate scores as |y - mu|/e^std
                numerator = np.abs(y - means)
                scores = numerator / np.e**stds

                # store scores per class
                for i, score in enumerate(scores):
                    pred_select = means[i].flatten()
                    scores = score.flatten()
                    # take scores clustered by predicted value separated into
                    # two clusters - positive and negative class
                    selected_pos = scores[pred_select >= 0.5]
                    selected_neg = scores[pred_select < 0.5]
                    if len(uncond_val) - 1 < i:
                        uncond_val.append(score.flatten())
                        cond_val.append([selected_pos, selected_neg])
                    else:
                        uncond_val[i] = np.concatenate([uncond_val[i],
                                                        score.flatten()])
                        cond_val[i][0] = np.concatenate([cond_val[i][0], selected_pos])
                        cond_val[i][1] = np.concatenate([cond_val[i][1], selected_neg])

        return uncond_val, cond_val

    def predict(self, inputs, significance=0.1):
        '''Generate prediction intervals for given batch and set significance
           level

        Parameters
        ----------
        inputs : Input tensor
        significance : Significance level, optional

        Returns
        -------
        Prediction intervals for the batch
        [classes, width, height, lower/upper bound], Mean prediction

        '''
        # perform MC dropout
        image = inputs.to(self._device, dtype=inputs.dtype)
        assert image.shape[0] == 1, 'Process one image at a time'

        means, stds = self._perform_mc_dropout(image)

        # extract scores
        uncond_scores = np.zeros((len(self.cal_scores_uncond), 1, 1))
        cond_scores_pos = np.zeros((len(self.cal_scores_cond), 1, 1))
        cond_scores_neg = np.zeros((len(self.cal_scores_cond), 1, 1))
        for i, cal_score in enumerate(self.cal_scores_uncond):
            idx = int(np.floor((1-significance) * len(cal_score)))
            uncond_scores[i] = cal_score[idx]

        for i, cal_score in enumerate(self.cal_scores_cond):
            idx_pos = int(np.floor((1-significance) * len(cal_score[0])))
            cond_scores_pos[i] = cal_score[0][idx_pos]
            idx_neg = int(np.floor((1-significance) * len(cal_score[1])))
            cond_scores_pos[i] = cal_score[1][idx_neg]

        # calculate upper and lower bound as mu + score_0.1 * e^std
        uncond_intervals = np.zeros((*means.shape, 2))
        uncond_bound = uncond_scores * np.e**stds
        cond_intervals = np.zeros((*means.shape, 2))
        # blend scores based on predicted value
        cond_scores = means * cond_scores_pos + (1 - means) * cond_scores_neg
        # scale scores based on difficulty
        cond_bound = cond_scores * np.e**stds

        # do not perform value clipping since we are actually most interested
        # in 2*bound, since we interpret this as uncertainty
        uncond_intervals[..., 0] = means - uncond_bound
        uncond_intervals[..., 1] = means + uncond_bound
        cond_intervals[..., 0] = means - cond_bound
        cond_intervals[..., 1] = means + cond_bound

        return uncond_intervals, cond_intervals, means

    def _perform_mc_dropout(self, image):
        '''Rum MC dropout for given image using batch_size and samples with
        which the ConformalRegressor was initialized

        Parameters
        ----------
        image : Image tensor [1, classes, height, width]

        Returns
        -------
        Dropout means [classes, height, width],
        Dropout std [classes, height, width]

        '''
        self._model._set_mc_dropout(True)

        mc_batch = image.repeat(self._batch_size, 1, 1, 1)
        repeats = self._samples // self._batch_size
        outputs = []
        for _ in range(repeats):
            predicted = self._model.proba(mc_batch)
            outputs.append(predicted)

        # estimate mean and std for each class and each pixel
        samples = np.concatenate(outputs)
        means = samples.mean(axis=0)
        stds = samples.std(axis=0)

        self._model._set_mc_dropout(False)

        return means, stds
