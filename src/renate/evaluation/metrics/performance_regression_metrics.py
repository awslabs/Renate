# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from torchmetrics import Metric

_DESCRIPTION = """
    In classification tasks, sample-wise inconsistencies between models appear as "negative flips":
    A new model incorrectly predicts the output for a test sample that was correctly
    classified by the old (reference) model. The fraction of the total number of negative flips
    and the total number of examples in the test set is called negative flip rate (NFR).
    Another metric for measuring regression is negative flip impact (NFI).
    Negative Flip Impact (NFI) measures the fraction of the total number of negative flips
    and the total number of errors in the test set. The difference between NFR and NFI
    is how they are normalized.
"""

_CITATION = """
    @article{Yan2021PositiveCongruentTT,
        title={Positive-Congruent Training: Towards Regression-Free Model Updates},
        author={Sijie Yan and Yuanjun Xiong and Kaustav Kundu and Shuo Yang
        and Siqi Deng and Meng Wang and Wei Xia and Stefano Soatto},
        journal={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2021},
        pages={14294-14303}
    }
    @inproceedings{Xie2021RegressionBugs,
        title={Regression Bugs Are In Your Model! Measuring, Reducing and Analyzing Regressions In
            NLP Model Updates},
        author={Yuqing Xie and Yi{-}An Lai and Yuanjun Xiong and Yi Zhang and Shuo Yang and Stefano
            Soatto},
        journal={Proceedings of the 59th Annual Meeting of the Association for Computational
            Linguistics and the 11th International Joint Conference on Natural Language Processing
            (Volume 1: Long Papers)},
        year={2021},
        pages={6589-6602}
        }
    @article{Cai2022MeasuringAR,
        title={Measuring and Reducing Model Update Regression in Structured Prediction for NLP},
        author={Deng Cai and Elman Mansimov and Yi-An Lai and Yixuan Su and Lei Shu and Yi Zhang},
        journal={ArXiv},
        year={2022},
        volume={abs/2202.02976}
    }
"""


class NegativeFlipRateMetric(Metric):
    """Compute Negative Flip Rate (NFR) between new and old models' predictions,

    NFR = len((pred_old == labels) and (pred_new != labels)) / len(test_set).
    """

    full_state_update: bool = False
    higher_is_better: bool = False
    is_differentiable: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("neg_flip_num", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, new_pred: torch.Tensor, old_pred: torch.Tensor, labels: torch.Tensor):
        """Update the metric.

        Args:
            new_pred: a 1-D torch tensor contains new model's predicted labels.
            old_pred: a 1-D torch tensor contains old model's predicted labels.
            labels: a 1-D torch tensor contains ground truth labels.
        """
        assert (
            new_pred.shape == old_pred.shape == labels.shape
        ), "dim of new_pred, old_pred, and labels are not matched!"
        assert len(new_pred) > 0, "input tensor is empty!"
        neg_flip_samples = (old_pred == labels) & (new_pred != labels)
        self.neg_flip_num += torch.sum(neg_flip_samples)
        self.total += len(labels)

    def compute(self):
        return self.neg_flip_num.float() / self.total


class NegativeFlipImpactMetric(Metric):
    """Compute Negative Flip Impact (NFI) between new and old models' predictions,

    NFI = len((pred_old == labels) and (pred_new != labels)) / len(pred_new != labels).
    """

    full_state_update: bool = False
    higher_is_better: bool = False
    is_differentiable: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("new_num_errors", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("neg_flip_num", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, new_pred: torch.Tensor, old_pred: torch.Tensor, labels: torch.Tensor):
        """Updates the metric.

        Args:
            new_pred: a 1-D torch tensor contains new model's predicted labels.
            old_pred: a 1-D torch tensor contains old model's predicted labels.
            labels: a 1-D torch tensor contains ground truth labels.
        """
        assert (
            new_pred.shape == old_pred.shape == labels.shape
        ), "dim of new_pred, old_pred, and labels are not matched!"
        assert len(new_pred) > 0, "input tensor is empty!"
        neg_flip_samples = (old_pred == labels) & (new_pred != labels)
        self.neg_flip_num += torch.sum(neg_flip_samples)
        self.new_num_errors += torch.sum(new_pred != labels)

    def compute(self):
        return self.neg_flip_num.float() / self.new_num_errors


class PositiveFlipRateMetric(Metric):
    """Compute Positive Flip Rate (PFR) between new and old models' predictions,

    PFR = len((pred_old != labels) and (pred_new == labels)) / len(test_set).
    """

    full_state_update: bool = False
    higher_is_better: bool = True
    is_differentiable: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("pos_flip_num", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, new_pred: torch.Tensor, old_pred: torch.Tensor, labels: torch.Tensor):
        """Updates the metric.

        Args:
            new_pred: a 1-D torch tensor contains new model's predicted labels.
            old_pred: a 1-D torch tensor contains old model's predicted labels.
            labels: a 1-D torch tensor contains ground truth labels.
        """
        assert (
            new_pred.shape == old_pred.shape == labels.shape
        ), "dim of new_pred, old_pred, and labels are not matched!"
        assert len(new_pred) > 0, "input tensor is empty!"
        pos_flip_samples = (old_pred != labels) & (new_pred == labels)
        self.pos_flip_num += torch.sum(pos_flip_samples)
        self.total += len(labels)

    def compute(self):
        return self.pos_flip_num.float() / self.total


class PositiveFlipImpactMetric(Metric):
    """Compute Positive Flip Impact (PFI) between new and old models' predictions,

    PFI = len((pred_old != labels) and (pred_new == labels)) / len(pred_new == labels).
    """

    full_state_update: bool = False
    higher_is_better: bool = True
    is_differentiable: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("new_num_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("pos_flip_num", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, new_pred: torch.Tensor, old_pred: torch.Tensor, labels: torch.Tensor):
        """Updates the metric.

        Args:
            new_pred: a 1-D torch tensor contains new model's predicted labels.
            old_pred: a 1-D torch tensor contains old model's predicted labels.
            labels: a 1-D torch tensor contains ground truth labels.
        """
        assert (
            new_pred.shape == old_pred.shape == labels.shape
        ), "dim of new_pred, old_pred, and labels are not matched!"
        assert len(new_pred) > 0, "input tensor is empty!"
        pos_flip_samples = (old_pred != labels) & (new_pred == labels)
        self.pos_flip_num += torch.sum(pos_flip_samples)
        self.new_num_correct += torch.sum(new_pred == labels)

    def compute(self):
        return self.pos_flip_num.float() / self.new_num_correct
