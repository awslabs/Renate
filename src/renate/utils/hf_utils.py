# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, Dict, List

from transformers import BatchEncoding, DataCollatorWithPadding


@dataclass
class DataCollatorWithPaddingForWildTime(DataCollatorWithPadding):
    """This adds to the `transformer` library's DataCollatorWithPadding. That data collator expects
    data in a standard HF format. Wild time data format are slightly different. We generally get a
    tuple of BatchEncoding (dict) and a class label. When being read from a buffer, an additional
    metadata attribute is present. These cases are not handled by the orig data collator.
    The code here only separates the input data into format original collator can handle and
    undoes the data packing: see parts after super().__call__.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first_element type determines if data sources
        first_element = features[0][0]
        to_be_collated = []
        assert len(features[0]) == 2, "Input to collator should contain only two elements"
        if isinstance(first_element, (BatchEncoding, dict)):
            # This is normal dataset and not a buffer with metadata, do the default things
            for elem in features:
                elem[0]["labels"] = elem[1]
                to_be_collated.append(elem[0])
            collated = super().__call__(to_be_collated)
            labels = collated.pop("labels")
            return collated, labels
        elif isinstance(first_element, tuple):
            # this has metadata possibly.
            for elem in features:
                elem[0][0]["metadata"] = elem[1] or False
                elem[0][0]["label"] = elem[0][1]
                to_be_collated.append(elem[0][0])
            collated = super().__call__(to_be_collated)
            metadata = collated.pop("metadata")
            if not metadata.any():
                metadata = {}
            labels = collated.pop("labels")
            return (collated, labels), metadata
        else:
            raise ValueError(f"Unknown structure to collate. Got {type(first_element[0])}")
