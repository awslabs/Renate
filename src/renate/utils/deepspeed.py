# Copyright 2020 The PyTorch Lightning team and Microsoft Corporation. All rights reserved.
# Modifications: Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved
# SPDX-License-Identifier: Apache-2.0

import os
import pickle as pkl
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from deepspeed.utils.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint,
    get_model_state_file,
    get_optim_files,
)

CPU_DEVICE = torch.device("cpu")


def ds_checkpoint_dir(checkpoint_dir: Union[str, Path], tag: Optional[str] = None) -> str:
    if tag is None:
        latest_path = os.path.join(checkpoint_dir, "latest")
        if os.path.isfile(latest_path):
            with open(latest_path) as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")

    directory = os.path.join(checkpoint_dir, tag)

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{ds_checkpoint_dir}' doesn't exist")
    return directory


# modified
def search_key(state: Dict[str, Any], substring: str) -> str:
    """This function looks for a substring in keys of dict and returns the full key that
    is the first match."""
    for k in state.keys():
        if substring in k:
            return k


# Modified script from
# https://github.com/Lightning-AI/lightning/blob/1.8.6/src/pytorch_lightning/utilities/deepspeed.py
# which is modified from
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/utils/zero_to_fp32.py
def convert_zero_checkpoint_to_fp32_state_dict(
    checkpoint_dir: Union[str, Path], tag: Optional[str] = None
) -> Dict[str, Any]:
    """Convert ZeRO 2 or 3 checkpoint into a single fp32 consolidated ``state_dict`` file
    that can be loaded with ``torch.load(file)`` + ``load_state_dict()`` and used for training
    without DeepSpeed.  Additionally the script has been modified to ensure we keep the
    lightning state inside the state dict for being able to run
    ``LightningModule.load_from_checkpoint('...')``.
    Modification to this version include the explicit handling of the _extra_state
    element of state dict. Deepspeed's and Lightning get-fp-32... functions only collate
    trainable parameters.

    Args:
        checkpoint_dir: path to the desired checkpoint folder.
            (one that contains the tag-folder, like ``global_step14``)
        tag: checkpoint tag used as a unique identifier for checkpoint. If not provided will
            attempt to load tag in the file named ``latest`` in the checkpoint folder,
            e.g., ``global_step14``.
    """

    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)
    # additional logic to ensure we keep the lightning state dict as well from rank 0.
    deepspeed_states = [
        "module",
        "optimizer",
        "lr_scheduler",
        "csr_tensor_module_names",
        "skipped_steps",
        "global_steps",
        "dp_world_size",
        "mp_world_size",
    ]
    checkpoint_dir = ds_checkpoint_dir(checkpoint_dir)
    optim_files = get_optim_files(checkpoint_dir)
    optim_state = torch.load(optim_files[0], map_location=CPU_DEVICE)
    zero_stage = optim_state["optimizer_state_dict"]["zero_stage"]
    model_file = get_model_state_file(checkpoint_dir, zero_stage)
    client_state = torch.load(model_file, map_location=CPU_DEVICE)
    # Assign extra_state by searching for which key it is
    extra_key = search_key(client_state["module"], "extra_state")
    extra_state = client_state["module"][extra_key]
    state_dict[extra_key] = extra_state
    # End of modifications
    client_state = {
        key: value for key, value in client_state.items() if key not in deepspeed_states
    }
    # State dict keys will include reference to wrapper _LightningModuleWrapperBase
    # Delete `module` prefix before saving.
    state_dict = {k.partition("module.")[2]: state_dict[k] for k in state_dict.keys()}
    client_state["state_dict"] = state_dict

    return client_state


def convert_to_tensor(obj):
    """This function converts a pickleable object to a torch tensor. This is only to
    aid saving with Deepspeed."""
    return torch.as_tensor(list(pkl.dumps(obj)))


def recover_object_from_tensor(tensor):
    """This function converts a tensor to a byte stream that is passed through pickle
    to recover the underlying object. For usage with Deepspeed"""
    return pkl.loads(bytes(tensor.tolist()))
