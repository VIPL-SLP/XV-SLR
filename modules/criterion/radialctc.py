import pdb
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from warpctc_pytorch import CTCLoss


class RadialCTC(nn.Module):
    def __init__(self, norm_scale=None, angle_margin=None, arc_margin=None, center_scale=None, nb_ratio=None,
                 blank_idx=0, reduction='sum', zero_infinity=False):
        super(RadialCTC, self).__init__()
        self.norm_scale = norm_scale
        self.center_scale = center_scale
        self.angle_margin = angle_margin
        self.arc_margin = arc_margin
        self.nb_ratio = nb_ratio
        self.ctc_utils = CTCHelper()
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction=reduction, zero_infinity=zero_infinity)
        # self.ctc_loss = CTCLoss(size_average=False, length_average=False)

    def forward(
            self, logits, labeling, logit_lgts, labeling_lgts,
            classifier_weight=None, feats=None, training_process=1.0,
    ):
        """
            Different from the orginial CTC implementation, for example, pytorch version
            (https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html?highlight=ctc#torch.nn.CTCLoss),
            RadialCTC adopts logits rather than log_probs as inputs for logits modification.
            :param logits: Tensor of size (T, N, C), the outputs of sequence recognition model.
            :param labeling: Tensor of size (N, S),targets are padded to the length of the longest sequence,
            and the target index cannot be blank (default=0).
            :param logit_lgts: Tuple or tensor of size (N), the lengths of the inputs (must each be <= T).
            And the lengths are specified for each sequence to achieve masking under the assumption that
            sequences are padded to equal lengths.
            :param labeling_lgts: Tuple or tensor of size (N), the lengths of the inputs (must each be <= S).
            And the lengths are specified for each sequence to achieve masking under the assumption that
            sequences are padded to equal lengths.
            :param classifier_weight: Classifier weight of size (D, C), which is used as the
            centroids of classes for constraint.
            :param feats: Frame-wise features of size (T * N, D)
            :param training_process:
            :return:
        """
        # Normalized weight matrix is used for anglular constraint and centralization
        normed_classifier_weight = F.normalize(classifier_weight, dim=0)

        # Decode maximal probability path for each sample.
        if self.center_scale or self.arc_margin or self.nb_ratio:
            max_prob_info = self.ctc_utils.decode_batch_max_path(
                (logits * self.norm_scale).log_softmax(dim=-1), logit_lgts, labeling, labeling_lgts
            )
            # max_prob_info2 = self.ctc_utils.decode_max_path(
            #     (logits * self.norm_scale).log_softmax(dim=-1), logit_lgts, labeling, labeling_lgts
            #     )
            # for res in zip(max_prob_info, max_prob_info2):
            #     for i in range(len(res[0])):
            #         print(len(res[0][i]), res[0][i])
            #         print(len(res[1][i]), res[1][i])
            #     pdb.set_trace()
        loss = 0
        if self.angle_margin is not None and self.angle_margin != 0:
            ang_loss = self.weight_angle_constraint(normed_classifier_weight)
            loss += ang_loss
        if self.center_scale is not None and self.center_scale != 0:
            center_loss = self.centralization_constraint(
                logits, feats,
                normed_classifier_weight,
                max_prob_info,
            )
            loss += center_loss
        if self.arc_margin or self.nb_ratio is not None:
            radial_loss = self.radial_constraint(
                logits, labeling, logit_lgts, labeling_lgts, max_prob_info, training_process, classifier_weight
            )
            loss += radial_loss
        else:
            logit_lgts = torch.IntTensor(logit_lgts).cpu()
            labeling_lgts = torch.IntTensor(labeling_lgts).cpu()
            # ctc_loss = self.ctc_loss((logits * self.norm_scale).cpu(), labeling.view(-1).int().cpu(), logit_lgts.int(),
            #                             labeling_lgts.int()).mean()
            ctc_loss = self.ctc_loss(
                (logits * self.norm_scale).log_softmax(dim=-1), labeling, logit_lgts.int(), labeling_lgts.int(),
            ).mean() / sum(labeling_lgts)
            loss += ctc_loss
        return loss

    def weight_angle_constraint(self, weight):
        # weight: (D, C) -> cosine: (C)
        cosine = torch.matmul(weight.T[0], weight)
        cols = len(cosine)
        # The self-cosine similarity of blank class should be one,
        # and the angle between blank class and other classes are constrained to the same value.
        constrained_angle = torch.ones(cols).to(cosine.device)
        constrained_angle[1:] *= math.cos(np.pi * self.angle_margin)
        loss = (((cosine - constrained_angle) * 1) ** 2)[1:].mean()
        if loss != loss:
            pdb.set_trace()
        return loss

    def centralization_constraint(self, logits, feats, centers, info):
        # logits: (T, N, C); feats: (T * N); centers: (D, C)
        # Select key frames with local maximum and pull them to class centroids
        ind_list, label_list = self.ctc_utils.keyframe_cal(logits, info, lambda x: x is not None, logits.shape[1])
        center = centers[:, label_list].T
        T, N, C = feats.shape
        feat = feats.reshape(T * N, C)[torch.hstack(ind_list).tolist()]
        # feat = feats.reshape(T * N, C)[ind_list]
        dist = (feat - center).pow(2).sum(dim=-1) * self.center_scale
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)
        if loss != loss:
            pdb.set_trace()
        return loss

    def radial_constraint(self, logits, labeling, logits_lgts, labeling_lgts, info, training_process, weight):
        # logits: (T, N, C); labeling: (N, S); logits_lgts: (N); labeling_lgts: (N)
        # ind_list, label_list = self.ctc_utils.keyframe_cal(logits, info, lambda x: x is not None, logits.shape[1])
        # Different from the original role of margin in face recognition, the margin here (m_hot) plays a role
        # of perturbation, which provides a different prediction for the key blank frame to release the overfitting.
        T, N, C = logits.shape
        if self.nb_ratio is not None:
            m_hot = torch.zeros(logits.shape, device=logits.device)
            acos_logits = torch.acos(logits)
            for i in range(N):
                lgt = logits_lgts[i]
                topk_idx = labeling_lgts[i] + int((lgt - labeling_lgts[i]) * self.nb_ratio) + 1
                blank_angle = acos_logits[:lgt, i, 0]
                label_list = [*set(labeling[i].tolist())]
                non_blank_angle = acos_logits[:lgt, i, label_list].min(dim=-1)[0]
                margin = -torch.topk(blank_angle - non_blank_angle, topk_idx, dim=0)[0][-2:].mean()
                m_hot[:, i, 0] += margin
            # cos(x + \theta) = cos(x)*cos(\theta) - sin(x)sin(\theta)
            new_logits = (logits * torch.cos(m_hot) - torch.sqrt(1 - logits ** 2) * torch.sin(m_hot))
            scaled_margin_logits = new_logits * self.norm_scale
        else:
            scaled_margin_logits = logits * self.norm_scale
        logits_lgts = torch.IntTensor(logits_lgts).cpu()
        labeling_lgts = torch.IntTensor(labeling_lgts).cpu()
        # margin_loss = self.ctc_loss(scaled_margin_logits.cpu(), labeling.view(-1).int().cpu(), logits_lgts.int(),
        #                             labeling_lgts.int()).mean()
        
        labeling_vec = labeling[labeling!=0]
        margin_loss = self.ctc_loss(scaled_margin_logits.log_softmax(-1), labeling_vec.view(-1).int().cpu(),
                                    logits_lgts.int(),
                                    labeling_lgts.int()).mean()
        margin_gradient = -(torch.autograd.grad(margin_loss, scaled_margin_logits, create_graph=True)[0]).detach()
        margin_label = scaled_margin_logits.softmax(dim=-1) + margin_gradient
        # np.save('margin_label.npy', margin_label.cpu().detach().numpy())
        # print(margin_label.shape)

        if self.arc_margin:
            splits_list = []
            for i in range(N):
                lab = margin_label[:, i].argmax(-1).tolist()
                splits = self.ctc_utils.generate_splits(lab)
                splits_list.append(splits)
            ind_list, label_list = self.ctc_utils.keyframe_cal(margin_label, splits_list,
                                                               lambda x: x is not None,
                                                               logits.shape[1])

            mat_index = torch.zeros(logits.shape, device=logits.device)
            mat_index.view(T * N, C)[ind_list, label_list] = 1
            # mat_index.scatter_(-1, hard_index[:, :, None], 1)
            mat_index[:, :, 0] = 0
            mat_index = mat_index.bool()
            arc_margin = self.angle_margin * self.angle_margin * np.pi
            cos_t = logits[mat_index]
            trunc_margin = torch.min(math.pi / 2 - torch.acos(cos_t), torch.ones_like(cos_t) * arc_margin)
            arc_logits = cos_t * torch.cos(trunc_margin) - torch.sqrt(1 - cos_t ** 2) * torch.sin(trunc_margin)
            cond = F.relu(arc_logits)
            keep = cos_t
            cos_t_add_m = torch.where(cond.bool(), arc_logits, keep)
            logits[mat_index] = cos_t_add_m
        new_scaled_margin_logits = logits * self.norm_scale
        ce_loss = -(new_scaled_margin_logits.log_softmax(-1) * margin_label.detach())
        # gradient = -(torch.autograd.grad(ce_loss.sum(), new_scaled_margin_logits, create_graph=True)[0]).detach()
        # gradient = -(torch.autograd.grad(ce_loss.sum(), weight, create_graph=True)[0]).detach()
        # if (gradient != gradient).any():
        #     pdb.set_trace()
        return ce_loss.sum() / sum(labeling_lgts)

    def cal_grad(self, loss, weight):
        return -(torch.autograd.grad(loss.sum(), weight, create_graph=True)[0]).detach()


class CTCHelper(object):
    def __init__(self):
        pass

    @staticmethod
    def torch2np(pytensor):
        if torch.is_tensor(pytensor):
            return pytensor.cpu().detach().numpy()
        elif isinstance(pytensor, np.ndarray):
            return pytensor
        else:
            assert "Wrong data type!"

    @staticmethod
    def log_sum_exp(inputs):
        a = max(inputs)
        prob_sum = 0
        for item in inputs:
            prob_sum += np.exp(item - a)
        return np.log(prob_sum) + a

    @staticmethod
    def decode_path(start, paths, label_list):
        decoded_path = list()
        decoded_path.append(start)
        for t in range(paths.shape[0] - 1, 0, -1):
            decoded_path.append(start - paths[t, start])
            start = decoded_path[-1]
        decoded_path = [label_list[idx] for idx in decoded_path[::-1]]
        return decoded_path

    @staticmethod
    def decode_batch_path(start, paths, label_list):
        batch, tlgt, clgt = paths.shape
        decoded_path = np.zeros((paths.shape[:2]), dtype=int)
        decoded_path[:, -1] = start
        for t in range(tlgt - 1, 0, -1):
            try:
                decoded_path[:, t - 1] = start - paths[np.arange(batch), t, start]
            except IndexError:
                pdb.set_trace()
            finally:
                pass
            start = decoded_path[:, t - 1]
        decoded_path = [label_list[i][decoded_path[i]].tolist() for i in range(batch)]
        return decoded_path

    def ctc_forward(self, logits, label, blank=0, operation="sum"):
        append_label = [label[i // 2] if i % 2 == 1 else blank for i in range(len(label) * 2 + 1)]
        len_lgt = len(logits)
        len_label = len(append_label)
        neginf = -1e8
        dp = np.ones((len_lgt, len_label)) * neginf
        paths = np.zeros((len_lgt, len_label), dtype=int)
        dp[0, 0] = logits[0, append_label[0]]
        dp[0, 1] = logits[0, append_label[1]]
        for t in range(1, len_lgt):
            for s in range(0, len_label):
                la1 = dp[t - 1, s]
                if s > 0:
                    la2 = dp[t - 1, s - 1]
                else:
                    la2 = neginf
                if s > 1 and append_label[s] != append_label[s - 2]:
                    la3 = dp[t - 1, s - 2]
                else:
                    la3 = neginf
                if operation == "sum":
                    dp[t, s] = self.log_sum_exp([la1, la2, la3]) + logits[t, append_label[s]]
                else:
                    dp[t, s] = max([la1, la2, la3]) + logits[t, append_label[s]]
                paths[t, s] = np.argmax([la1, la2, la3])
        if operation == "sum":
            return dp, append_label, len_lgt, len_label
        else:
            return dp, append_label, len_lgt, len_label, paths

    def batch_ctc_forward(self, logits, logits_lgt, label, label_lgt, blank=0, operation="sum"):
        batch_size = logits.shape[1]

        append_label_lgt = label_lgt * 2 + 1
        batch_append_label = np.zeros((label.shape[0], append_label_lgt.max().item()), dtype=int)
        batch_append_label[:, 1::2] = label

        temporal_lgt = len(logits)
        max_append_label_lgt = append_label_lgt.max().item()
        neginf = -1e8
        dp = np.ones((batch_size, temporal_lgt, max_append_label_lgt)) * neginf
        paths = np.zeros((batch_size, temporal_lgt, max_append_label_lgt), dtype=int)
        dp[:, 0, 0] = logits[0, np.arange(batch_size), batch_append_label[:, 0]]
        dp[:, 0, 1] = logits[0, np.arange(batch_size), batch_append_label[:, 1]]
        for t in range(1, temporal_lgt):
            for s in range(0, max_append_label_lgt):
                la1 = dp[:, t - 1, s]
                if s > 0:
                    la2 = dp[:, t - 1, s - 1]
                else:
                    la2 = np.ones((batch_size,)) * neginf
                if s > 1:
                    la3 = dp[:, t - 1, s - 2] * (batch_append_label[:, s] != batch_append_label[:, s - 2]) + \
                          np.ones((batch_size,)) * neginf * (batch_append_label[:, s] == batch_append_label[:, s - 2])
                else:
                    la3 = np.ones((batch_size,)) * neginf
                if operation == "sum":
                    dp[t, s] = self.log_sum_exp([la1, la2, la3]) + logits[t, append_label[s]]
                else:
                    dp[:, t, s] = np.maximum(np.maximum(la1, la2), la3) + logits[
                        t, np.arange(batch_size), batch_append_label[:, s]]
                paths[:, t, s] = np.argmax(np.vstack([la1, la2, la3]), axis=0)
        if operation == "sum":
            return dp, append_label, len_lgt, len_label
        else:
            return dp, batch_append_label, temporal_lgt, append_label_lgt, paths

    @staticmethod
    def generate_splits(decoded_path):
        st_idx, ed_idx = 0, 0
        splits = list()
        for idx, lab in enumerate(decoded_path):
            if lab == decoded_path[st_idx]:
                ed_idx += 1
            else:
                splits.append([decoded_path[st_idx], st_idx, ed_idx])
                st_idx = idx
                ed_idx = idx + 1
        if st_idx < len(decoded_path):
            splits.append([decoded_path[st_idx], st_idx, ed_idx])
        assert sum([item[2] - item[1] for item in splits]) == len(decoded_path), f"{max_path}, {splits}"
        return splits

    def decode_max_path(self, log_probs, logits_lgt, labels, label_lgt):
        batch_size = log_probs.shape[1]
        path_list = []
        splits_list = []
        for sample_idx in range(batch_size):
            sample_probs = self.torch2np(log_probs[:logits_lgt[sample_idx], sample_idx])
            sample_labels = labels[sample_idx].tolist()[:label_lgt[sample_idx]]
            dp_max_mat, label_list, lgt, lgt_label, paths = \
                self.ctc_forward(sample_probs, sample_labels, blank=0, operation="max")
            start = lgt_label - 1 - np.argmax(
                [dp_max_mat[lgt - 1, lgt_label - 1], dp_max_mat[lgt - 1, lgt_label - 2]])
            max_path = self.decode_path(start, paths, label_list)
            splits = self.generate_splits(max_path)
            path_list.append(max_path)
            splits_list.append(splits)
        return path_list, splits_list

    def decode_batch_max_path(self, log_probs, logits_lgt, labels, label_lgt):
        if isinstance(logits_lgt, list):
            logits_lgt = np.array(logits_lgt, dtype=int)
        if isinstance(label_lgt, list):
            label_lgt = np.array(label_lgt, dtype=int)

        batch_size = log_probs.shape[1]
        splits_list = []
        dp_max_mat, label_list, lgt, lgt_label, paths = \
            self.batch_ctc_forward(self.torch2np(log_probs), logits_lgt, self.torch2np(labels), label_lgt, blank=0,
                                   operation="max")
        start_idx = lgt_label - 1 - np.argmax(
            np.vstack([
                dp_max_mat[np.arange(batch_size), logits_lgt - 1][np.arange(batch_size), lgt_label - 1],
                dp_max_mat[np.arange(batch_size), logits_lgt - 1][np.arange(batch_size), lgt_label - 2],
            ]), axis=0
        )
        path_list = self.decode_batch_path(start_idx, paths, label_list)
        for sample_idx in range(batch_size):
            splits = self.generate_splits(path_list[sample_idx])
            splits_list.append(splits)
        return path_list, splits_list

    @staticmethod
    def keyframe_cal(logits, info, lambda_func, batch_multiplier=1):
        ind_list = []
        label_list = []
        split_list = info[1] if isinstance(info, tuple) else info
        for idx, splits in enumerate(split_list):
            ind_list += [(logits[item[1]:item[2], idx, item[0]].argmax() + item[1]) * batch_multiplier + idx for item in
                         [*filter(lambda_func, splits)]]
            label_list += [item[0] for item in [*filter(lambda_func, splits)]]
        return ind_list, label_list


if __name__ == '__main__':
    sample = np.load('/home/ycmin/example.npy', allow_pickle=True).item()
    metric = RadialCTC(norm_scale=16, angle_margin=0.4, arc_margin=0.3, center_scale=0.1)
    # print(sample)
    loss = metric(sample['logits'], sample['labeling'], sample['logit_lgts'], sample['labeling_lgts'],
                  classifier_weight=sample['classifier_weight'], feats=sample['feats'].cpu())
    pdb.set_trace()
    # info = np.load('examples.npy', allow_pickle=True).item()
    # print(info.keys())
    # ctc_utils = CTCHelper()
    # import time
    #
    # st = time.time()
    # max_prob_info1 = ctc_utils.decode_max_path(
    #     (info['logits'] * 64).log_softmax(dim=-1), info['logit_lgts'], info['labeling'], info['labeling_lgts']
    # )
    # print(time.time() - st)
    # st = time.time()
    # max_prob_info2 = ctc_utils.decode_batch_max_path(
    #     (info['logits'] * 64).log_softmax(dim=-1), info['logit_lgts'], info['labeling'], info['labeling_lgts']
    # )
    # print(time.time() - st)
    # for idx in range(len(max_prob_info1[0])):
    #     assert max_prob_info1[0][idx] == max_prob_info2[0][idx]
    #     assert max_prob_info1[1][idx] == max_prob_info2[1][idx]
    