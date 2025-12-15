import numpy as np
import torch


class MeanIoUEvaluator:
    def __init__(self, num_classes, class_names, has_bg=True, ignore_index=0):
        self.num_classes = num_classes + int(has_bg)
        self.class_names = class_names
        self.tp = [0] * self.num_classes
        self.fp = [0] * self.num_classes
        self.fn = [0] * self.num_classes
        self.ignore_index = ignore_index

    @torch.no_grad()
    def update(self, pred, gt):
        valid = (gt != self.ignore_index)

        for i_part in range(0, self.num_classes):
            tmp_gt = (gt == i_part)
            tmp_pred = (pred == i_part)
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & valid).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & valid).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & valid).item()

    def reset(self):
        self.tp = [0] * self.num_classes
        self.fp = [0] * self.num_classes
        self.fn = [0] * self.num_classes

    def return_score(self, verbose=True, name='dataset', suppress_prints=False):
        jac = [0] * self.num_classes
        for i_part in range(self.num_classes):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        eval_result = dict()
        eval_result['jaccards_all_categs'] = jac
        eval_result['mIoU'] = np.mean(jac)

        if not suppress_prints or verbose:
            print(f'Evaluation for segmentation - {name}')
            print('mIoU is %.2f' % (100*eval_result['mIoU']))
            if verbose:
                for i_part in range(self.num_classes):
                    print('IoU class %s is %.2f' % (self.class_names[i_part], 100 * jac[i_part]))

        return eval_result

    def __str__(self):
        res = self.return_score(verbose=False, suppress_prints=True)['mIoU']*100
        fmtstr = "IoU ({0:.2f})"
        return fmtstr.format(res)