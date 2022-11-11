# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from argparse import Namespace

import pytest

from renate.cli.parsing_functions import _get_args_by_prefix, parse_unknown_args


def test_parse_unknown_args_with_malformed_input():
    with pytest.raises(ValueError, match=r"Error: unable to parse the additional arguments..*"):
        parse_unknown_args(["--arg_with_no_value_associated"])

    with pytest.raises(ValueError, match=r"Please make sure arguments are specified.*"):
        parse_unknown_args(["not_starting_with--arg1", "val1"])


@pytest.mark.parametrize(
    "unknown_args_list, expected_result",
    [([], {}), (["--arg1", "value1", "--arg2", "value2"], {"arg1": "value1", "arg2": "value2"})],
    ids=["no-unknown-args-exist", "valid-input"],
)
def test_parse_unknown_args_with_valid_input(unknown_args_list, expected_result):
    assert parse_unknown_args(unknown_args_list) == expected_result


@pytest.mark.parametrize(
    "kwargs, prefix, expected_args",
    [
        (
            {"other_hp": 2, "model_fn_hp1": 1, "model_fn_hp2": "v"},
            "model_fn_",
            {"model_fn_hp1": 1, "model_fn_hp2": "v"},
        ),
        ({"other_hp": 2, "model_fn_hp1": 1, "model_fn_hp2": "v"}, "data_module_fn_", {}),
    ],
    ids=["non-empty-subset", "empty-result"],
)
@pytest.mark.parametrize("use_namespace", [True, False], ids=["Namespace", "Dict"])
def test_get_args_by_prefix(kwargs, prefix, expected_args, use_namespace):
    """Tests if function correctly returns all keys starting with the given `prefix`."""
    args = Namespace(**kwargs) if use_namespace else kwargs
    assert _get_args_by_prefix(args, prefix) == expected_args
