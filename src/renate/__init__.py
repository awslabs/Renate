# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging

_root_logger = logging.getLogger()
_renate_logger = logging.getLogger(__name__)
_renate_logger.setLevel(logging.INFO)

if not _root_logger.hasHandlers():
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s")
    )
    _renate_logger.addHandler(_handler)
    _renate_logger.propagate = False

__version__ = "0.4.0"
