# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import math
from typing import List, Optional, Set, Tuple, Union

import torch
from torch.utils.data import Dataset, random_split
from transformers import BatchEncoding

from renate import defaults
from renate.types import NestedTensors


def reinitialize_model_parameters(model: torch.nn.Module) -> None:
    """Reinitializes the parameters of a model.

    This relies on all submodules of `model` implementing a method `reset_parameters()`. This
    is implemented for core `torch.nn` layers, but this may not be the case for custom
    implementations of exotic layers. A warning is logged for modules that do not implement
    `reset_parameters()`.

    The actual logic of renitializing parameters depends on the type of layer. It may affect the
    module's buffers (non-trainable parameters, e.g., batch norm stats) as well.

    Args:
        model: The model to be re-initialized.
    """
    for module in model.modules():
        # Skip modules without any parameters of their own.
        if not list(module.parameters(recurse=False)) and not list(module.buffers(recurse=False)):
            continue
        try:
            module.reset_parameters()
        except AttributeError:
            logging.warning(f"Skipping module {module} while resetting parameters.")


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """Provides a torch.Generator for the given seed.

    torch.default_generator is returned if seed is None.
    """
    if seed is None:
        return torch.default_generator
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    return rng


def randomly_split_data(
    dataset: Dataset, proportions: List[float], seed: int = defaults.SEED
) -> List[Dataset]:
    """Randomly splits a dataset into chunks."""
    rng = get_generator(seed)
    split_sizes = _proportions_into_sizes(proportions, len(dataset))
    return random_split(dataset, split_sizes, generator=rng)


def _proportions_into_sizes(proportions: List[float], size: int) -> List[int]:
    """A helper function to convert chunk proportions into sizes.

    In case of rounding doubts, any remaining samples are appended to the last split size.
    """
    assert math.isclose(sum(proportions), 1.0), sum(proportions)
    sizes = [round(proportion * size) for proportion in proportions[:-1]]
    sizes.append(size - sum(sizes))
    return sizes


def move_tensors_to_device(tensors: NestedTensors, device: torch.device) -> NestedTensors:
    """Moves a collection of tensors to `device`.

    The collection `tensors` can be a nested structure of tensors, tuples, lists, and dicts.
    """
    if isinstance(tensors, (BatchEncoding, torch.Tensor)):
        return tensors.to(device)
    elif isinstance(tensors, tuple):
        return tuple(move_tensors_to_device(t, device) for t in tensors)
    # We need to include lists here as well, since collate_fn sometimes turns tuples into lists.
    # See https://github.com/pytorch/pytorch/issues/48419.
    elif isinstance(tensors, list):
        return [move_tensors_to_device(t, device) for t in tensors]
    elif isinstance(tensors, dict):
        return {key: move_tensors_to_device(t, device) for key, t in tensors.items()}
    else:
        raise TypeError(
            "Expected `tensors` to be a nested structure of tensors, tuples, list and dict; "
            f"discovered {type(tensors)}."
        )


def get_length_nested_tensors(batch: NestedTensors) -> torch.Size:
    """Given a NestedTensor, return its length.

    Assumes that the first axis in each element is the same.
    """
    if isinstance(batch, torch.Tensor):
        return batch.shape[0]
    if isinstance(batch, tuple):
        return batch[0].shape[0]
    if isinstance(batch, dict):
        return batch[next(iter(batch.keys()))].shape[0]


def cat_nested_tensors(
    nested_tensors: Union[Tuple[NestedTensors], List[NestedTensors]], axis: int = 0
) -> NestedTensors:
    """Concatenates the two NestedTensors.

    Equivalent of PyTorch's ``cat`` function for ``NestedTensors``.

    Args:
        nested_tensors: Tensors to be concatenated.
        axis: Concatenation axis.
    """
    if isinstance(nested_tensors[0], torch.Tensor):
        return torch.cat(nested_tensors, axis)
    if isinstance(nested_tensors[0], tuple):
        return tuple(
            cat_nested_tensors(nested_tensor, axis) for nested_tensor in zip(*nested_tensors)
        )
    if isinstance(nested_tensors[0], dict):
        return {
            key: cat_nested_tensors([nested_tensor[key] for nested_tensor in nested_tensors], axis)
            for key in nested_tensors[0]
        }


def unique_classes(dataset: torch.utils.data.Dataset) -> Set[int]:
    """Compute the unique class ids in a dataset.

    Args:
        dataset: Instance of Torch dataset.
    """
    from renate.memory.buffer import DataBuffer  # to avoid circular import

    if isinstance(dataset, DataBuffer):
        label_element = lambda elem: elem[0][1]
    else:
        label_element = lambda elem: elem[1]

    unique_values = set()
    for ind in range(len(dataset)):
        label = label_element(dataset[ind])
        unique_values.add(label.item())

    return unique_values


def complementary_indices(num_outputs: int, valid_classes: Set[int]) -> List[int]:
    """Compute the asymmetric difference between the two arguments

    Args:
        num_outputs: An integer of total number of classes the model can output.
        valid_classes: A set of integers of valid classes.
    """
    return [class_idx for class_idx in range(num_outputs) if class_idx not in valid_classes]
