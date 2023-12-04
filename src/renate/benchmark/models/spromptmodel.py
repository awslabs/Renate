# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import math
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from renate.models.layers.shared_linear import SharedMultipleLinear
from renate.models.prediction_strategies import PredictionStrategy
from renate.models.task_identification_strategies import TaskPrototypes

from .l2p import PromptedTransformer
from .base import RenateBenchmarkingModule

logger = logging.getLogger(__name__)


class PromptPool(nn.Module):
    """Implements a pool of prompts to be used in for S-Prompts.

    Args:
        prompt_size: Equivalent to number of input tokens used per update . Defaults to 10.
        embedding_size: Hidden size of the transformer used.. Defaults to 768.
        current_update_id: Current update it. Used to init number of prompts. Defaults to 0.
    """

    def __init__(
        self, prompt_size: int = 10, embedding_size: int = 768, current_update_id: int = 0
    ) -> None:
        super().__init__()
        self._M = prompt_size
        self._embedding_size = embedding_size
        self._curr_task = current_update_id

        self._pool = nn.ParameterDict()
        for id in range(self._curr_task):
            # This always needs to be intialized as the torch's state dict only restores for an
            # existing Parameter.
            self._pool[f"{id}"] = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty((self._M, self._embedding_size)), a=math.sqrt(5)
                )
            )

        self._pool.requires_grad_(True)

    def forward(self, id: int) -> torch.nn.Parameter:
        return self._pool[f"{id}"]

    def get_params(self, id: int) -> List[torch.nn.Parameter]:
        return [self._pool[f"{id}"]]

    def increment_task(self) -> None:
        self._pool[f"{len(self._pool)}"] = nn.Parameter(
            torch.empty((self._M, self._embedding_size)).uniform_(-1, 1)
        )
        self._pool.requires_grad_(True)


class SPromptTransformer(RenateBenchmarkingModule):
    """Implements Transformer Model for S-Prompts as described in  Wang, Yabin, et.al ."S-prompts
       learning with pre-trained transformers: An occam’s razor for domain incremental learning."
       Advances in Neural Information Processing Systems 35 (2022): 5682-5695.

    Args:
        pretrained_model_name_or_path: A string that denotes which pretrained model from the HF hub
            to use.
        image_size: Image size. Used if `pretrained_model_name_or_path` is not set .
        patch_size: Patch size to be extracted. Used if `pretrained_model_name_or_path` is not set .
        num_layers: Num of transformer layers. Used only if `pretrained_model_name_or_path` is not
            set .
        num_heads: Num heads in MHSA. Used only if `pretrained_model_name_or_path` is not set .
        hidden_dim: Hidden dimension of transformers. Used only if `pretrained_model_name_or_path`
            is not set .
        mlp_dim: _description_. Used only if `pretrained_model_name_or_path` is not set .
        dropout: _description_. Used only if `pretrained_model_name_or_path` is not set .
        attention_dropout: _description_. Used only if `pretrained_model_name_or_path` is not set .
        num_outputs: Number of output classes of the output. Defaults to 10.
        prediction_strategy: Continual learning strategies may alter the prediction at train or test
            time. Defaults to None.
        add_icarl_class_means: If ``True``, additional parameters used only by the
            ``ICaRLModelUpdater`` are added. Only required when using that updater.
        prompt_size: Equivalent to number of input tokens used per update . Defaults to 10.
        task_id: Internal variable used to increment update id. Shouldn't be set by user.
            Defaults to 0.
        clusters_per_task: Number clusters in k-means used for task identification. Defaults to 5.
        per_task_classifier: Flag to share or use a common classifier head for all tasks.
            Defaults to False.
    """

    def __init__(
        self,
        pretrained_model_name_or_path="google/vit-base-patch16-224",
        image_size: int = 32,
        patch_size: int = 4,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 768,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        num_outputs: int = 10,
        prediction_strategy: Optional[PredictionStrategy] = None,
        add_icarl_class_means: bool = True,
        prompt_size: int = 10,
        task_id: int = 0,
        clusters_per_task: int = 5,
        per_task_classifier: bool = False,
    ):
        transformer = PromptedTransformer(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_outputs=num_outputs,
            prediction_strategy=prediction_strategy,
            add_icarl_class_means=add_icarl_class_means,
        )
        super().__init__(
            embedding_size=transformer.transformer._embedding_size,
            num_outputs=num_outputs,
            constructor_arguments=dict(
                **transformer.transformer._constructor_arguments,
                prompt_size=prompt_size,
                task_id=task_id,
                clusters_per_task=clusters_per_task,
                per_task_classifier=per_task_classifier,
            ),
            prediction_strategy=prediction_strategy,
            add_icarl_class_means=add_icarl_class_means,
        )
        self._M = prompt_size
        self._task_id = task_id
        self._per_task_classifier = per_task_classifier

        prompt_pool = PromptPool(
            prompt_size=self._M,
            embedding_size=self._embedding_size,
            current_update_id=self._task_id,
        )

        self._backbone = nn.ModuleDict({"transformer": transformer, "prompt_pool": prompt_pool})
        self._task_id_method = TaskPrototypes(
            task_id=task_id,
            clusters_per_task=clusters_per_task,
            embedding_size=self._embedding_size,
        )
        self._backbone["transformer"].requires_grad_(False)
        self._backbone["prompt_pool"].requires_grad_(True)

        self._backbone["classifier"] = SharedMultipleLinear(
            self._embedding_size,
            self._num_outputs,
            share_parameters=not self._per_task_classifier,
            num_updates=self._task_id + 1,
        )

        self._tasks_params = nn.ModuleDict(
            {k: nn.Identity() for k, _ in self._tasks_params.items()}
        )

        self._backbone.forward = self._forward_for_monkey_patching
        self.task_ids = None

    def increment_task(self) -> None:
        # This cannot be a part of add_task_params as the super.__init__ function calls
        # add_task_params, and thus we would be trying parameters to the non-existent
        # self.s_prompts
        self._backbone["prompt_pool"].increment_task()

    def _forward_for_monkey_patching(
        self, x: Union[torch.Tensor, Dict[str, Any]], task_id: str = None
    ) -> torch.Tensor:
        prompt = None
        task_ids = None
        if self.training:
            prompt = self._backbone["prompt_pool"](self._task_id)
        else:
            task_ids = self._task_id_method.infer_task(self._backbone["transformer"](x))
            if task_ids is not None:
                prompt = torch.stack([self._backbone["prompt_pool"](i) for i in task_ids])
                self.task_ids = task_ids.detach().cpu().numpy()

        features = self._backbone["transformer"](x, prompt)

        # additional logic for separate classifiers
        # a. This forward returns logits directly, and the RenateBenchmarkingModule's _task_params
        #    now are identities. Thus, the overall operation is still the network forward pass.
        # b. Additional handling of params is not needed as backbone's params will return all the
        #    necessary elements.

        if self.training:
            logits = self._backbone["classifier"][f"{self._task_id}"](features)
        elif task_ids is not None:
            logits = torch.cat(
                [
                    self._backbone["classifier"][f"{t}"](feat.unsqueeze(0))
                    for t, feat in zip(task_ids, features)
                ]
            )
        else:
            logits = self._backbone["classifier"]["0"](features)

        return logits

    def update_task_identifier(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self._task_id_method.update_task_prototypes(features, labels)

    def set_extra_state(self, state: Any, decode=True):
        super().set_extra_state(state, decode)
        # once this is set (after loading. increase that by one.)
        self._constructor_arguments["task_id"] = self._task_id + 1

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone["transformer"](x)
