# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import pytest

from renate.cli.parsing_functions import (
    CUSTOM_ARGS_GROUP,
    REQUIRED_ARGS_GROUP,
    get_argument_type,
    get_data_module_fn_kwargs,
    get_function_args,
    get_metrics_fn_kwargs,
    get_model_fn_kwargs,
    to_dense_str,
)
from renate.utils.module import import_module

config_file = str(Path(__file__).parent.parent / "renate_config_files" / "config.py")
config_module = import_module("config_module", config_file)


def test_get_argument_type():
    """Test if correct argument type is extracted from typed function."""

    def foo(
        int_param: int,
        float_list_param: List[float],
        tuple_param: tuple,
        dict_param: Dict[str, int],
        union_param: Union[str, int],
        list_of_float_lists: List[List[float]],
        no_annotation_param,
        optional_str_param: Optional[str] = None,
        optional_dict_param: Optional[Dict[str, str]] = None,
    ):
        pass

    arg_spec = inspect.getfullargspec(foo)
    expected_types = {
        "int_param": int,
        "float_list_param": list,
        "tuple_param": tuple,
        "optional_str_param": str,
        "list_of_float_lists": list,
    }
    expected_errors = {
        "dict_param": r"Type typing.Dict\[str, int\] is not supported \(argument dict_param\).",
        "union_param": r"Type typing.Union\[str, int\] is not supported \(argument union_param\).",
        "optional_dict_param": r"Type typing.Union\[typing.Dict\[str, str\], NoneType\] is not "
        r"supported \(argument optional_dict_param\).",
        "no_annotation_param": r"Missing type annotation for argument no_annotation_param.",
    }
    if sys.version_info.minor >= 9:
        expected_errors["optional_dict_param"] = (
            r"Type typing.Optional\[typing.Dict\[str, str\]\] is not supported "
            r"\(argument optional_dict_param\)."
        )
    for argument_name, expected_type in expected_types.items():
        assert get_argument_type(arg_spec=arg_spec, argument_name=argument_name) == expected_type

    for argument_name, expected_error in expected_errors.items():
        with pytest.raises(TypeError, match=expected_error):
            get_argument_type(arg_spec=arg_spec, argument_name=argument_name)


@pytest.mark.parametrize(
    "all_args,ignore_args",
    [
        ({}, []),
        ({}, ["val_size", "seed"]),
        (
            {
                "existing_arg": {
                    "type": float,
                    "argument_group": REQUIRED_ARGS_GROUP,
                    "required": True,
                }
            },
            ["data_path"],
        ),
    ],
    ids=("No all_args, no ignore_args", "no all_args", "all_args"),
)
def test_get_function_args(all_args, ignore_args):
    expected_args = [
        "data_path",
        "val_size",
        "seed",
        "groupings",
        "optional_tuple",
        "optional_float",
        "list_param",
        "bool_param",
    ]
    expected_all_args = {
        **all_args,
        **{
            "data_path": {
                "type": str,
                "argument_group": CUSTOM_ARGS_GROUP,
                "required": True,
                "true_type": str,
            },
            "val_size": {
                "type": float,
                "argument_group": CUSTOM_ARGS_GROUP,
                "default": 0.0,
                "true_type": float,
            },
            "seed": {
                "type": int,
                "argument_group": CUSTOM_ARGS_GROUP,
                "default": 0,
                "true_type": int,
            },
            "groupings": {
                "type": str,
                "argument_group": CUSTOM_ARGS_GROUP,
                "default": "((0,1),(2,3,4))",
                "true_type": tuple,
            },
            "optional_tuple": {
                "type": str,
                "argument_group": CUSTOM_ARGS_GROUP,
                "default": None,
                "true_type": tuple,
            },
            "optional_float": {
                "type": float,
                "argument_group": CUSTOM_ARGS_GROUP,
                "default": None,
                "true_type": float,
            },
            "list_param": {
                "type": str,
                "argument_group": CUSTOM_ARGS_GROUP,
                "default": "[1,2]",
                "true_type": list,
            },
            "bool_param": {
                "type": str,
                "argument_group": CUSTOM_ARGS_GROUP,
                "default": "False",
                "true_type": bool,
            },
        },
    }
    for arg in ignore_args:
        del expected_all_args[arg]
    args = get_function_args(
        config_module=config_module,
        function_name="data_module_fn",
        all_args=all_args,
        ignore_args=ignore_args,
    )
    assert args == expected_args
    assert all_args == expected_all_args


def test_get_function_args_with_inconsistent_args():
    """If types of arguments are different, an error should be raised."""
    all_args = {
        "model_state_url": {
            "type": float,
            "argument_group": REQUIRED_ARGS_GROUP,
            "required": True,
        }
    }

    with pytest.raises(
        TypeError,
        match=r"Types of `model_state_url` are not consistent. Defined as type `<class 'str'>` "
        r"as well as `<class 'float'>`.",
    ):
        get_function_args(
            config_module=config_module,
            function_name="model_fn",
            all_args=all_args,
            ignore_args=[],
        )


def test_get_function_args_prefers_required_true():
    """If argument is defined multiple times, it is required if it is required only once."""
    all_args = {
        "data_path": {
            "type": str,
            "argument_group": REQUIRED_ARGS_GROUP,
            "required": False,
        }
    }
    get_function_args(
        config_module=config_module,
        function_name="data_module_fn",
        all_args=all_args,
        ignore_args=[],
    )
    assert all_args["data_path"]["required"]


def test_get_fn_kwargs_helper_functions():
    """Tests whether the different helper functions correctly create kwargs given a dictionary
    and the Python function."""
    expected_data_module_kwargs = {
        "data_path": "home/data/path",
        "groupings": ((1, 2), (3, 4)),
        "optional_float": None,
        "bool_param": False,
    }
    config_space = {
        "data_path": expected_data_module_kwargs["data_path"],
        "model_state_url": "home/model/state",
        "unused_config": 1,
        "groupings": to_dense_str(expected_data_module_kwargs["groupings"]),
        "optional_float": to_dense_str(expected_data_module_kwargs["optional_float"]),
        "bool_param": to_dense_str(expected_data_module_kwargs["bool_param"]),
        "num_outputs": 10,
    }
    data_module_kwargs = get_data_module_fn_kwargs(
        config_module=config_module, config_space=config_space, cast_arguments=True
    )
    assert data_module_kwargs == expected_data_module_kwargs
    model_kwargs = get_model_fn_kwargs(
        config_module=config_module, config_space=config_space, cast_arguments=True
    )
    assert model_kwargs == {"model_state_url": config_space["model_state_url"]}
    metrics_kwargs = get_metrics_fn_kwargs(
        config_module=config_module, config_space=config_space, cast_arguments=True
    )
    assert metrics_kwargs == {"num_outputs": config_space["num_outputs"]}
