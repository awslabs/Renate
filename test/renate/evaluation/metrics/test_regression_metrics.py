# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from renate.evaluation.metrics.performance_regression_metrics import (
    NegativeFlipImpactMetric,
    NegativeFlipRateMetric,
    PositiveFlipImpactMetric,
    PositiveFlipRateMetric,
)


@pytest.fixture
def input_tensor():
    # nfr = 1 / 4 = 0.25
    # pfr = 1 / 4 = 0.25
    # nfi = 1 / 2 = 0.5
    # pfi = 1 / 2 = 0.5
    new_pred = torch.tensor([1, 3, 0, 3])
    old_pred = torch.tensor([2, 1, 0, 2])
    gt = torch.tensor([3, 3, 0, 2])
    return new_pred, old_pred, gt


@pytest.fixture
def input_numpy_array():
    np_new_pred = np.asarray([1, 3, 0, 3])
    np_old_pred = np.asarray([2, 1, 0, 2])
    np_gt = np.asarray([3, 3, 0, 2])
    return np_new_pred, np_old_pred, np_gt


def test_nfr_metric(input_tensor, input_numpy_array):
    new_pred, old_pred, gt = input_tensor
    np_new_pred, np_old_pred, np_gt = input_numpy_array
    manual_nfr = 0.25

    NFR_metric = NegativeFlipRateMetric()

    correct_old_model = np.equal(np_old_pred, np_gt)
    error_new_model = np.not_equal(np_new_pred, np_gt)
    num_negative_flips = np.logical_and(correct_old_model, error_new_model).sum()

    nfr = num_negative_flips / float(len(gt))

    assert NFR_metric(new_pred, old_pred, gt) == nfr
    assert NFR_metric(new_pred, old_pred, gt) == manual_nfr


def test_pfr_metric(input_tensor, input_numpy_array):
    new_pred, old_pred, gt = input_tensor
    np_new_pred, np_old_pred, np_gt = input_numpy_array
    manual_pfr = 0.25

    PFR_metric = PositiveFlipRateMetric()

    correct_new_model = np.equal(np_new_pred, np_gt)
    error_old_model = np.not_equal(np_old_pred, np_gt)
    num_positive_flips = np.logical_and(correct_new_model, error_old_model).sum()

    pfr = num_positive_flips / float(len(gt))

    assert PFR_metric(new_pred, old_pred, gt) == pfr
    assert PFR_metric(new_pred, old_pred, gt) == manual_pfr


def test_nfi_metric(input_tensor, input_numpy_array):
    new_pred, old_pred, gt = input_tensor
    np_new_pred, np_old_pred, np_gt = input_numpy_array
    manual_nfi = 0.5

    NFI_metric = NegativeFlipImpactMetric()

    correct_old_model = np.equal(np_old_pred, np_gt)
    error_new_model = np.not_equal(np_new_pred, np_gt)
    num_negative_flips = np.logical_and(correct_old_model, error_new_model).sum()

    nfi = num_negative_flips / float(error_new_model.sum())

    assert NFI_metric(new_pred, old_pred, gt) == nfi
    assert NFI_metric(new_pred, old_pred, gt) == manual_nfi


def test_pfi_metric(input_tensor, input_numpy_array):
    new_pred, old_pred, gt = input_tensor
    np_new_pred, np_old_pred, np_gt = input_numpy_array
    manual_pfi = 0.5

    PFI_metric = PositiveFlipImpactMetric()

    correct_new_model = np.equal(np_new_pred, np_gt)
    error_old_model = np.not_equal(np_old_pred, np_gt)
    num_positive_flips = np.logical_and(correct_new_model, error_old_model).sum()

    pfi = num_positive_flips / float(correct_new_model.sum())

    assert PFI_metric(new_pred, old_pred, gt) == pfi
    assert PFI_metric(new_pred, old_pred, gt) == manual_pfi


def test_regression_metric_muti_batch(input_tensor):
    new_pred, old_pred, gt = input_tensor
    new_pred_2 = torch.tensor([3, 1, 1, 3])

    NFR_metric = NegativeFlipRateMetric()
    NFR_metric.update(new_pred, old_pred, gt)
    NFR_metric.update(new_pred_2, old_pred, gt)

    manual_nfr_multi_batch = 0.375

    assert NFR_metric.compute() == manual_nfr_multi_batch


def test_regression_metric_corner_case_empty_input():
    new_pred = torch.tensor([])
    old_pred = torch.tensor([])
    gt = torch.tensor([])

    NFR_metric = NegativeFlipRateMetric()

    with pytest.raises(AssertionError):
        NFR_metric(new_pred, old_pred, gt)


def test_regression_metric_corner_case_unmatch_input():
    new_pred = torch.tensor([1, 2])
    old_pred = torch.tensor([1])
    gt = torch.tensor([1])

    NFR_metric = NegativeFlipRateMetric()

    with pytest.raises(AssertionError):
        NFR_metric(new_pred, old_pred, gt)
