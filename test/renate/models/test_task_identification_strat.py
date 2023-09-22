# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from sklearn.cluster import KMeans

from renate.models.task_identification_strategies import TaskPrototypes


def test_task_prototypes():
    data = torch.nn.functional.normalize(torch.rand(10, 3))
    labels = torch.arange(start=0, end=data.size(0))
    task_proto = TaskPrototypes(0, 0, data.size(1))
    # lets attach
    task_proto._training_feat_centroids = data
    task_proto._training_feat_task_ids = labels

    test_data = torch.nn.functional.normalize(torch.rand(5, 3))
    predictions = task_proto.infer_task(test_data)

    kmeans = KMeans(n_clusters=data.size(0))
    kmeans.cluster_centers_ = data.numpy()
    kmeans.labels_ = labels.numpy()
    kmeans._n_threads = 1

    gnd_truth = kmeans.predict(test_data.numpy())

    assert (predictions.numpy() == gnd_truth).all()
