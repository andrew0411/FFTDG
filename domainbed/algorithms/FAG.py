import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from domainbed.optimizers import get_optimizer


def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)

class FAG(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(FAG, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)


        if self.update_count <= self.hparams.fag_iters:
            penalty_weight = self.hparams.fag_penalty
        else:
            penalty_weight = 1.0


        losses = torch.zeros(len(minibatches))

        cnt = 0
        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            
            lam = np.random.beta(self.hparams.beta_mix, self.hparams.beta_mix)
            print(xi.size())
            L = np.random.uniform(0.1, 0.25)
            aug_x = self.get_aug_x(xi, xj, L, lam)

            print(aug_x.size())


            prediction = self.predict(aug_x)
            objective = F.cross_entropy(prediction, yi)
            losses[cnt] = objective
            cnt += 1
        
        mean = losses.mean()
        penalty = ((losses-mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams.fag_iters:
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss':loss.item(), 'penalty': penalty}


    def get_aug_x(self, xi, xj, L, lam):
        xi_amp, xi_pha = self.extract(xi)
        xj_amp, xj_pha = self.extract(xj)

        b, c, h, w = xi.size()
        crops = (np.floor( np.amin((h, w)) * L)).astype(int)

        xi_amp[:, :, 0:crops, 0:crops] = lam*xi_amp[:, :, 0:crops, 0:crops] + (1-lam)*xj_amp[:, :, 0:crops, 0:crops]
        xi_amp[:, :, 0:crops, w-crops:w] = lam*xi_amp[:, :, 0:crops, w-crops:w] + (1-lam)*xj_amp[:, :, 0:crops, w-crops:w]
        xi_amp[:, :, h-crops:h, 0:crops] = lam*xi_amp[:, :, h-crops:h, 0:crops] + (1-lam)*xj_amp[:, :, h-crops:h, 0:crops]
        xi_amp[:, :, h-crops:h, w-crops:w] = lam*xi_amp[:, :, h-crops:h, w-crops:w] + (1-lam)*xj_amp[:, :, h-crops:h, w-crops:w]

        aug_ = torch.zeros((b, c, h, w, 2), dtype=torch.float)
        aug_[:, :, :, :, 0] = torch.cos(xi_pha) * xi_amp
        aug_[:, :, :, :, 1] = torch.sin(xi_pha) * xi_amp

        aug_x = torch.irfft(aug_, signal_ndim=2, onesided=False, signal_sizes=[h, w])

        return aug_x.cuda()

    def extract(self, x):
        # x : [32, 3, 224, 224]
        x_fft = torch.rfft(x, signal_ndim=2, onesided=False)

        # x_fft : [32, 3, 224, 224, 2]
        fft_amp = x_fft[:, :, :, :, 0]**2 + x_fft[:, :, :, :, 1]**2
        fft_amp = torch.sqrt(fft_amp)
        fft_pha = torch.atan2(x_fft[:, :, :, :, 1], x_fft[:, :, :, :, 0])

        # fft_amp, fft_pha = [32, 3, 224, 224]
        return fft_amp, fft_pha