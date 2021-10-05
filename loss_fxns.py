import torch
from torch import nn
import torch.nn.functional as F


class DiscriminativeLoss(nn.Module):
    """ This class computes the loss
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, device, delta_dist=3.0):
        """ Create the criterion.
        Parameters:
        """
        super().__init__()
        self.delta_dist = delta_dist
        self.param_var = 1.0
        self.param_dist = 1.0
        self.param_reg = 0.0001
        self.device = device
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def instance_loss(self, outputs, targets):
        embedding = outputs
        correct_label = targets
        # flatten
        _, feature_dim, _, _ = embedding.shape
        embedding = embedding.view(feature_dim, -1).transpose(0, 1)  # pixels x features
        _, num_instances, _, _ = correct_label.shape
        correct_label = correct_label.view(num_instances, -1)  # instances x pixels
        # add bg instance
        bg = torch.unsqueeze((torch.clip(torch.sum(correct_label, dim=0), 0., 1.) - 1) * (-1), 0)  # 1 x pixels
        label_list = list(torch.split(correct_label, dim=0, split_size_or_sections=1)) + [bg]
        correct_label = torch.cat(label_list, dim=0)  # instances+1 x pixels
        num_instances += 1
        # calculate mean embedding
        counts = torch.sum(correct_label, dim=1)  # instances
        mu = torch.matmul(correct_label, embedding) / torch.unsqueeze((counts + 1e-8), 1)  # instances x feature_dim
        # calculates losses
        l_var = self._lvar(correct_label, mu, num_instances, embedding)
        # _ldist throws errors and makes no sense for <2 instances, therefore it's set to zero in this case
        if num_instances > 1:
            l_dist = self._ldist(mu, num_instances, feature_dim)
        else:
            l_dist = 0.0
        l_reg = torch.mean(torch.norm(mu, dim=1)) * self.param_reg
        disc_loss = l_var + l_dist + l_reg
        loss_dict = {
            'l_var': l_var,
            'l_dist': l_dist,
            'l_reg': l_reg,
            'disc_loss': disc_loss
        }
        return disc_loss, loss_dict

    def _ldist(self, mu, num_instances, feature_dim):
        # Get L1-,distance for each pair of clusters like this:
        #   mu_1 - mu_1
        #   mu_1 - mu_2
        #   mu_1 - mu_3
        mu = mu.transpose(0, 1).unsqueeze(2).expand(feature_dim, num_instances,
                                                    num_instances)  # feature_dim x instances x instances
        mu_band_rep = mu.reshape(feature_dim, num_instances * num_instances)  # feature_dim x instances*instances
        mu_interleaved_rep = mu.permute(0, 2, 1).reshape(feature_dim,
                                                         num_instances * num_instances)  # feature_dim x instances*instances
        mu_diff = mu_band_rep - mu_interleaved_rep  # feature_dim x instances*instances
        mu_dist = torch.norm(mu_diff, dim=0)
        mask = torch.logical_not(mu_dist.eq(0.0))
        mu_dist = torch.masked_select(mu_dist, mask)
        mu_dist = 2 * self.delta_dist - mu_dist  # apply hinge
        mu_dist = F.relu(mu_dist)  # remove the ones below the hinge
        mu_dist = torch.square(mu_dist)
        l_dist = torch.mean(mu_dist)
        return l_dist

    def _lvar(self, correct_label, mu, num_instances, embedding):
        # l_var
        mu_expand = torch.matmul(correct_label.transpose(0, 1), mu)  # pixels x feature_dim
        counts = torch.sum(correct_label, dim=1)  # instances+1
        distance = torch.norm(mu_expand - embedding, dim=1, keepdim=True)  # 1 x pixels
        distance = torch.square(distance)  # 1 x pixels
        l_var = torch.squeeze(torch.matmul(correct_label, distance))  # instance + 1
        l_var = l_var / (counts + 1e-8)
        l_var = torch.sum(l_var)
        l_var = l_var / (num_instances + 1e-8)
        return l_var

    def forward(self, input, target):
        # split
        pred_fgbg = input[:, 0:1, :, :]
        pred_emb = input[:, 1:, :, :]
        #
        acc_loss = []
        b = input.shape[0]
        if b > 1:
            for idx in range(b):
                tmp_target = F.one_hot(target.long())
                tmp_target = target[idx]  # 1 x H x W
                tmp_target = torch.permute(F.one_hot(tmp_target.long()), (0, 3, 1, 2))
                _, tmp_target = torch.unique(tmp_target, return_inverse=True)  # make labels consecutive numbers
                inst_loss, loss_dict = self.instance_loss(pred_emb[idx].unsqueeze(0), tmp_target.float())
                acc_loss.append(inst_loss)
            inst_loss = torch.mean(torch.stack(acc_loss))
        else:
            _, tmp_target = torch.unique(target, return_inverse=True)
            tmp_target = torch.permute(F.one_hot(tmp_target.squeeze(0).long()), (0, 3, 1, 2))
            inst_loss, loss_dict = self.instance_loss(pred_emb, tmp_target.float())
        # prepare fgbg target
        target_fgbg = (target > 0).float()
        fg_bg_loss = self.bce_loss(input=pred_fgbg, target=target_fgbg)
        return inst_loss + fg_bg_loss