# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Union


def int_or_str(x: str) -> Union[str, int]:
    """Function to cast to int or str.

    This is used to tackle precision which can be int (16, 32) or str (bf16)
    """
    try:
        return int(x)
    except ValueError:
        return x
