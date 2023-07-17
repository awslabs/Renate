# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import torch
from torch.utils.data import default_collate
from transformers import BatchEncoding, DataCollatorWithPadding


@dataclass
class DataCollatorWithPaddingForWildTime(DataCollatorWithPadding):
    """A data collator class that can handle wild time data (non-standard) batches.

    This adds to the `transformer` library's DataCollatorWithPadding. That data collator expects
    data in a standard HF format. Wild time data format is slightly different: We get a
    tuple of BatchEncoding (dict) and a class label. When being read from a buffer, an additional
    metadata attribute is present. These cases are not handled by the orig data collator.
    The code here only separates the input data into format original collator can handle and
    undoes the data packing: see parts after super().__call__.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first_element type determines if data sources
        # known types : {features}, class    (from wildtimedataset)
        #             : ({features}, class), metadata (from a replay buffer : val data?)
        #             : index, ({features}, class) (by _EnumeratedDataset in BaseER traindata)
        # Note that {features} can be a dict or BatchEncoding.
        first_element = features[0][0]
        assert len(features[0]) == 2, "Input to collator should contain only two elements"
        if isinstance(first_element, (BatchEncoding, dict)):
            # This is normal dataset and not a buffer with metadata, do the default things
            return self._collate_hf_class_tuples(features)
        elif isinstance(first_element, tuple):
            # this has metadata possibly.
            collated, labels = self._collate_hf_class_tuples([elem[0] for elem in features])
            metadata_to_collate = [elem[1] for elem in features]
            collated_metadata = default_collate(metadata_to_collate)
            return (collated, labels), collated_metadata
        elif isinstance(first_element, (torch.Tensor, int)):
            ## this is EnumeratedDataset.
            collated_indices = default_collate([elem[0] for elem in features])
            collated, labels = self._collate_hf_class_tuples([elem[1] for elem in features])
            return collated_indices, (collated, labels)

        else:
            raise ValueError(
                f"Unknown structure to collate. Got {first_element} of {type(first_element)}"
            )

    def _collate_hf_class_tuples(
        self, features: Tuple[Union[BatchEncoding, Dict[str, Any]], Union[torch.Tensor, int]]
    ):
        to_be_collated = []
        for elem in features:
            elem[0]["labels"] = elem[1]
            to_be_collated.append(elem[0])
        collated = super().__call__(to_be_collated)
        labels = collated.pop("labels")
        return collated, labels
