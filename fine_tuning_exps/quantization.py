#### This is from https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/quantization.py
from contextlib import contextmanager
import torch
import bitsandbytes as bnb  # noqa: E402


class Linear8bitLt(bnb.nn.Linear8bitLt):
    """Wraps `bnb.nn.Linear8bitLt` and enables instantiation directly on the device and
    re-quantizaton when loading the state dict.


    This should only be used for inference. For training, use `bnb.nn.Linear8bitLt` directly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, has_fp16_weights=False, threshold=6.0)
        # We quantize the initial weight here so we don't end up filling the device
        # memory with float32 weights which could lead to OOM.
        self._quantize_weight(self.weight.data)

    def _load_from_state_dict(self, local_state_dict, *args, **kwargs):
        # There is only one key that ends with `*.weight`, the other one is the bias
        weight_key = next(
            (name for name in local_state_dict.keys() if name.endswith("weight")),
            None,
        )
        if weight_key is None:
            return

        # Load the weight from the state dict and re-quantize it
        weight = local_state_dict.pop(weight_key)
        self._quantize_weight(weight)

        # If there is a bias, let nn.Module load it
        if local_state_dict:
            super()._load_from_state_dict(local_state_dict, *args, **kwargs)

    def _quantize_weight(self, weight: torch.Tensor) -> None:
        # This code is taken and adapted from `bnb.nn.Int8Params.cuda()`
        B = weight.contiguous().half().cuda()
        CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
        del CBt
        del SCBt
        self.weight.data = CB
        setattr(self.weight, "CB", CB)
        setattr(self.weight, "SCB", SCB)


@contextmanager
def quantization(mode: str = None):
    quantized_linear_cls = None
    if mode == "llm.int8":
        quantized_linear_cls = bnb.nn.Linear8bitLt
    elif mode is not None:
        raise ValueError(f"mode={mode} is unknown. Use llm.int8 or None.")
    enabled = mode is not None
    torch_linear_cls = torch.nn.Linear
    if enabled:
        torch.nn.Linear = quantized_linear_cls
    yield
    if enabled:
        torch.nn.Linear = torch_linear_cls
