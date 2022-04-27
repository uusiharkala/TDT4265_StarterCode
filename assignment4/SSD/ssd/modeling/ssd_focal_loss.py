import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np

def focal_loss(confs, y, gamma, alpha):
        y_one_hot = F.one_hot(y, 9)
        y_one_hot_a = torch.transpose(y_one_hot, 1, 2).contiguous()
        a = torch.pow(1 - F.softmax(confs, dim=1), gamma)
        b = F.log_softmax(confs, dim=1)
        loss = torch.sum(- alpha * a * y_one_hot_a * b, dim=1)
        return loss

class SSDFocalLoss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels using focal loss
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """

    def __init__(self, anchors):
        super().__init__()
        self.scale_xy = 1.0 / anchors.scale_xy
        self.scale_wh = 1.0 / anchors.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.anchors = nn.Parameter(anchors(order="xywh").transpose(0, 1).unsqueeze(dim=0),
                                    requires_grad=False)
        self.gamma = 2
        self.alpha = torch.Tensor([0.01, 1, 1, 1, 1, 1, 1, 1, 1]).to("cuda:0")
        self.alpha = self.alpha.view(1, -1, 1)

    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy * (loc[:, :2, :] - self.anchors[:, :2, :]) / self.anchors[:, 2:, ]
        gwh = self.scale_wh * (loc[:, 2:, :] / self.anchors[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()


    def forward(self,
                bbox_delta: torch.FloatTensor, confs: torch.FloatTensor,
                gt_bbox: torch.FloatTensor, gt_labels: torch.LongTensor):
        """
        NA is the number of anchor boxes (by default this is 8732)
            bbox_delta: [batch_size, 4, num_anchors]
            confs: [batch_size, num_classes, num_anchors]
            gt_bbox: [batch_size, num_anchors, 4]
            gt_label = [batch_size, num_anchors]
        """

        gt_bbox = gt_bbox.transpose(1, 2).contiguous()  # reshape to [batch_size, 4, num_anchors]

        classification_loss = focal_loss(confs, gt_labels, self.gamma, self.alpha)
        classification_loss = classification_loss.mean()

        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
        bbox_delta = bbox_delta[pos_mask]
        gt_locations = self._loc_vec(gt_bbox)
        gt_locations = gt_locations[pos_mask]
        regression_loss = F.smooth_l1_loss(bbox_delta, gt_locations, reduction="sum")
        num_pos = gt_locations.shape[0] / 4
        total_loss = regression_loss / num_pos + classification_loss
        to_log = dict(
            regression_loss=regression_loss / num_pos,
            classification_loss=classification_loss / num_pos,
            total_loss=total_loss
            )
        #print("Classification Loss: " + str(to_log["classification_loss"]))
        #print("Total Loss: " + str(to_log["total_loss"]))
        return total_loss, to_log
