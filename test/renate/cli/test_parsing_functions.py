# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import inspect
from typing import Dict, List, Optional, Union

import pytest

from renate.cli.parsing_functions import get_attribute_type


def test_get_attribute_type():
    def foo(
        int_param: int,
        float_list_param: List[float],
        tuple_param: tuple,
        dict_param: Dict[str, int],
        union_param: Union[str, int],
        optional_str_param: Optional[str] = None,
    ):
        pass

    arg_spec = inspect.getfullargspec(foo)
    expected_types = {
        "int_param": int,
        "float_list_param": list,
        "tuple_param": tuple,
        "optional_str_param": str,
    }
    expected_errors = {
        "dict_param": r"Type typing.Dict\[str, int\] is not supported \(Attribute dict_param\).",
        "union_param": r"Type typing.Union\[str, int\] is not supported \(Attribute union_param\).",
    }
    for attribute_name, expected_type in expected_types.items():
        assert get_attribute_type(arg_spec=arg_spec, attribute_name=attribute_name) == expected_type

    for attribute_name, expected_error in expected_errors.items():
        with pytest.raises(TypeError, match=expected_error):
            get_attribute_type(arg_spec=arg_spec, attribute_name=attribute_name)
