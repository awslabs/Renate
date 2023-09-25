# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torchmetrics
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Subset

from renate import defaults
from renate.data.datasets import _EnumeratedDataset, _TransformedDataset
from renate.memory.buffer import DataDict
from renate.models import RenateModule
from renate.types import NestedTensors
from renate.updaters.learner import ReplayLearner
from renate.updaters.learner_components.component import Component
from renate.updaters.learner_components.losses import (
    WeightedCLSLossComponent,
    WeightedCustomLossComponent,
    WeightedMeanSquaredErrorLossComponent,
    WeightedPooledOutputDistillationLossComponent,
)
from renate.updaters.learner_components.reinitialization import (
    ShrinkAndPerturbReinitializationComponent,
)
from renate.updaters.model_updater import SingleTrainingLoopUpdater
from renate.utils.misc import maybe_populate_mask_and_ignore_logits
from renate.utils.pytorch import move_tensors_to_device


class BaseExperienceReplayLearner(ReplayLearner, abc.ABC):
    """A base implementation of experience replay.

    It is designed for the online CL setting, where only one pass over each new chunk of data is
    allowed. The Learner maintains a Reservoir buffer. In the training step, it samples a batch of
    data from the memory and appends it to the batch of current-task data. At the end of the
    training step, the memory is updated.

    Args:
        components: An ordered dictionary of components that are part of the experience replay
            learner.
        loss_weight: A scalar weight factor for the base loss function to trade it off with other
            loss functions added by `components`.
        ema_memory_update_gamma: The gamma used for exponential moving average to update the meta
            data with respect to the logits and intermediate representation, if there is some.
        loss_normalization: Whether to normalize the loss by the weights of all the components.
    """

    def __init__(
        self,
        components: Dict[str, Component],
        loss_weight: float = defaults.LOSS_WEIGHT,
        ema_memory_update_gamma: float = defaults.EMA_MEMORY_UPDATE_GAMMA,
        loss_normalization: int = defaults.LOSS_NORMALIZATION,
        **kwargs: Any,
    ) -> None:
        self._components_names = list(components.keys())
        super().__init__(**kwargs)
        self._memory_loader: Optional[DataLoader] = None
        self._components = components
        self._loss_weight = loss_weight
        self._ema_memory_update_gamma = ema_memory_update_gamma
        self._use_loss_normalization = bool(loss_normalization)

    def _create_metrics_collections(
        self, logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None
    ) -> None:
        super()._create_metrics_collections(logged_metrics)
        for name in self._components_names:
            if name in self._loss_collections:
                raise ValueError(
                    f"Component name {name} is already used as a loss name. Please pick a "
                    "different name."
                )
            self._loss_collections["train_losses"].update({name: torchmetrics.MeanMetric()})

    def on_model_update_start(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        train_dataset_collate_fn: Optional[Callable] = None,
        val_dataset_collate_fn: Optional[Callable] = None,
        task_id: Optional[str] = None,
    ) -> None:
        """Called before a model update starts."""
        super().on_model_update_start(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataset_collate_fn=train_dataset_collate_fn,
            val_dataset_collate_fn=val_dataset_collate_fn,
            task_id=task_id,
        )
        self._set_memory_loader()

    def train_dataloader(self) -> DataLoader:
        train_dataset = _EnumeratedDataset(
            _TransformedDataset(
                self._train_dataset,
                transform=self._train_transform,
                target_transform=self._train_target_transform,
            )
        )
        return DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            generator=self._rng,
            pin_memory=True,
            collate_fn=self._train_collate_fn,
        )

    def on_train_start(self) -> None:
        """PyTorch Lightning function to be run at the start of the training."""
        super().on_train_start()
        for component in self._components.values():
            component.on_train_start(model=self._model)

    def training_step(
        self, batch: Tuple[torch.Tensor, Tuple[NestedTensors, torch.Tensor]], batch_idx: int
    ) -> STEP_OUTPUT:
        """PyTorch Lightning function to return the training loss."""
        idx, (inputs, targets) = batch
        step_output = super().training_step(batch=(inputs, targets), batch_idx=batch_idx)
        step_output["train_data_idx"] = idx
        step_output["loss"] *= self._loss_weight

        batch_memory: Optional[torch.Tensor] = None
        metadata_memory: Optional[torch.Tensor] = None
        outputs_memory: Optional[torch.Tensor] = None
        intermediate_representation_memory: Optional[List[torch.Tensor]] = None
        loss_normalization = self._loss_weight
        if self._memory_loader is not None:
            for name, component in self._components.items():
                memory_sampled = False
                if component.sample_new_memory_batch or batch_memory is None:
                    batch_memory = self._sample_from_buffer(device=step_output["loss"].device)
                    (inputs_memory, _), metadata_memory = batch_memory
                    outputs_memory = self(inputs_memory)

                    outputs_memory, self._class_mask = maybe_populate_mask_and_ignore_logits(
                        self._mask_unused_classes,
                        self._class_mask,
                        self._classes_in_current_task,
                        outputs_memory,
                    )
                    intermediate_representation_memory = (
                        self._model.get_intermediate_representation()
                    )
                    self._model.reset_intermediate_representation_cache()
                    memory_sampled = True

                component_loss = component.loss(
                    outputs_memory=outputs_memory,
                    batch_memory=batch_memory,
                    intermediate_representation_memory=intermediate_representation_memory,
                ).mean()
                self._loss_collections["train_losses"][name](component_loss)
                step_output["loss"] += component_loss
                loss_normalization += component.weight
                if memory_sampled and self._ema_memory_update_gamma < 1.0:
                    mem_idx = metadata_memory["idx"].cpu()
                    self._memory_buffer.metadata["outputs"][mem_idx] = self._memory_buffer.metadata[
                        "outputs"
                    ][
                        mem_idx
                    ].cpu() * self._ema_memory_update_gamma + outputs_memory.detach().cpu() * (
                        1.0 - self._ema_memory_update_gamma
                    )
            if self._use_loss_normalization:
                step_output["loss"] /= loss_normalization

        return step_output

    def _sample_from_buffer(self, device: torch.device) -> Optional[Tuple[NestedTensors, DataDict]]:
        """Function to sample from the buffer, if buffer is populated."""
        if self._memory_loader is not None and len(self._memory_buffer) >= self._memory_batch_size:
            memory_batch = next(iter(self._memory_loader))
            return move_tensors_to_device(memory_batch, device)
        else:
            return None

    def training_step_end(self, step_output: STEP_OUTPUT) -> STEP_OUTPUT:
        """PyTorch Lightning function to perform after the training step."""
        super().training_step_end(step_output)
        self._update_memory_buffer(step_output)
        return step_output

    def _update_memory_buffer(self, step_output: STEP_OUTPUT) -> None:
        outputs = step_output["outputs"]
        metadata = {"outputs": outputs.detach().cpu()}
        for i, intermediate_representation in enumerate(step_output["intermediate_representation"]):
            metadata[
                f"intermediate_representation_{i}"
            ] = intermediate_representation.detach().cpu()
        # Some datasets have problems using tensors as subset indices, convert to list of ints.
        train_data_idx = [int(idx) for idx in step_output["train_data_idx"]]
        dataset = Subset(self._train_dataset, train_data_idx)
        self._memory_buffer.update(dataset, metadata)
        self._set_memory_loader()

    def _set_memory_loader(self) -> None:
        """Create a memory loader from a memory buffer."""
        if self._memory_loader is None and len(self._memory_buffer) >= self._memory_batch_size:
            self._memory_loader = DataLoader(
                dataset=self._memory_buffer,
                batch_size=self._memory_batch_size,
                drop_last=True,
                shuffle=True,
                generator=self._rng,
                pin_memory=True,
                collate_fn=self._train_collate_fn,
            )

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """PyTorch Lightning function to perform after the training and optimizer step."""
        super().on_train_batch_end(outputs, batch, batch_idx)
        for component in self._components.values():
            component.on_train_batch_end(model=self._model)

    @abc.abstractmethod
    def components(self, **kwargs) -> Dict[str, Component]:
        """Returns the components of the learner.

        This is a user-defined function that should return a dictionary of components.
        """

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load states of components."""
        super().on_load_checkpoint(checkpoint)
        for component in self._components.values():
            component.on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save states of components."""
        super().on_save_checkpoint(checkpoint)
        for component in self._components.values():
            component.on_save_checkpoint(checkpoint)


class ExperienceReplayLearner(BaseExperienceReplayLearner):
    """This is the version of experience replay proposed in

    Chaudhry, Arslan, et al. "On tiny episodic memories in continual learning."
    arXiv preprint arXiv:1902.10486 (2019).

    Args:
        alpha: The weight of the cross-entropy loss component applied to the memory samples.
    """

    def __init__(self, alpha: float = defaults.ER_ALPHA, **kwargs) -> None:
        components = self.components(loss_fn=kwargs["loss_fn"], alpha=alpha)
        super().__init__(components=components, **kwargs)

    def components(
        self, loss_fn: Optional[torch.nn.Module] = None, alpha: float = defaults.ER_ALPHA
    ) -> Dict[str, Component]:
        return {
            "memory_loss": WeightedCustomLossComponent(
                loss_fn=loss_fn, weight=alpha, sample_new_memory_batch=True
            )
        }


class DarkExperienceReplayLearner(ExperienceReplayLearner):
    """A Learner that implements Dark Experience Replay.

    Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara:
    Dark Experience for General Continual Learning: a Strong, Simple Baseline. NeurIPS 2020

    Args:
        alpha: The weight of the mean squared error loss component between memorised logits and the
            current logits on the memory data.
        beta: The weight of the cross-entropy loss component between memorised targets and the
            current logits on the memory data.
    """

    def __init__(
        self, alpha: float = defaults.DER_ALPHA, beta: float = defaults.DER_BETA, **kwargs
    ) -> None:
        super().__init__(alpha=beta, **kwargs)
        self._components = self.components(loss_fn=kwargs["loss_fn"], alpha=alpha, beta=beta)

    def components(
        self,
        loss_fn: Optional[torch.nn.Module] = None,
        alpha: float = defaults.DER_ALPHA,
        beta: float = defaults.DER_BETA,
    ) -> Dict[str, Component]:
        components = super().components(loss_fn=loss_fn, alpha=beta)
        components.update(
            {
                "mse_loss": WeightedMeanSquaredErrorLossComponent(
                    weight=alpha, sample_new_memory_batch=False
                )
            }
        )
        return components


class PooledOutputDistillationExperienceReplayLearner(BaseExperienceReplayLearner):
    """A Learner that implements Pooled Output Distillation.

    Douillard, Arthur, et al. "Podnet: Pooled outputs distillation for small-tasks incremental
    learning."
    European Conference on Computer Vision. Springer, Cham, 2020.

    Args:
        alpha: Scaling value which scales the loss with respect to all intermediate representations.
        distillation_type: Which distillation type to apply with respect to the intermediate
            representation.
        normalize: Whether to normalize both the current and cached features before computing the
            Frobenius norm.
    """

    def __init__(
        self,
        alpha: float = defaults.POD_ALPHA,
        distillation_type: str = defaults.POD_DISTILLATION_TYPE,
        normalize: bool = defaults.POD_NORMALIZE,
        **kwargs,
    ) -> None:
        components = self.components(
            alpha=alpha, distillation_type=distillation_type, normalize=normalize
        )
        super().__init__(components=components, **kwargs)

    def components(
        self,
        alpha: float = defaults.POD_ALPHA,
        distillation_type: str = defaults.POD_DISTILLATION_TYPE,
        normalize: bool = defaults.POD_NORMALIZE,
    ) -> Dict[str, Component]:
        return {
            "pod_loss": WeightedPooledOutputDistillationLossComponent(
                weight=alpha,
                sample_new_memory_batch=True,
                distillation_type=distillation_type,
                normalize=normalize,
            )
        }


class CLSExperienceReplayLearner(BaseExperienceReplayLearner):
    """A learner that implements a Complementary Learning Systems Based Experience Replay.

    Arani, Elahe, Fahad Sarfraz, and Bahram Zonooz.
    "Learning fast, learning slow: A general continual learning method based on complementary
    learning system."
    arXiv preprint arXiv:2201.12604 (2022).

    Args:
        alpha: Scaling value for the cross-entropy loss.
        beta: Scaling value for the consistency loss.
        stable_model_update_weight: The starting weight for the exponential moving average to update
            the stable model copy.
        plastic_model_update_weight: The starting weight for the exponential moving average to
            update the plastic model copy.
        stable_model_update_probability: The probability to update the stable model copy.
        plastic_model_update_probability: The probability to update the plastic model copy.
    """

    def __init__(
        self,
        alpha: float = defaults.CLS_ALPHA,
        beta: float = defaults.CLS_BETA,
        stable_model_update_weight: float = defaults.CLS_STABLE_MODEL_UPDATE_WEIGHT,
        plastic_model_update_weight: float = defaults.CLS_PLASTIC_MODEL_UPDATE_WEIGHT,
        stable_model_update_probability: float = defaults.CLS_STABLE_MODEL_UPDATE_PROBABILITY,
        plastic_model_update_probability: float = defaults.CLS_PLASTIC_MODEL_UPDATE_PROBABILITY,
        **kwargs,
    ):
        components = self.components(
            model=kwargs["model"],
            loss_fn=kwargs["loss_fn"],
            alpha=alpha,
            beta=beta,
            stable_model_update_weight=stable_model_update_weight,
            plastic_model_update_weight=plastic_model_update_weight,
            stable_model_update_probability=stable_model_update_probability,
            plastic_model_update_probability=plastic_model_update_probability,
        )

        super().__init__(components=components, **kwargs)

    def components(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        alpha: float = defaults.CLS_ALPHA,
        beta: float = defaults.CLS_BETA,
        plastic_model_update_weight: float = defaults.CLS_PLASTIC_MODEL_UPDATE_WEIGHT,
        stable_model_update_weight: float = defaults.CLS_STABLE_MODEL_UPDATE_WEIGHT,
        plastic_model_update_probability: float = defaults.CLS_PLASTIC_MODEL_UPDATE_PROBABILITY,
        stable_model_update_probability: float = defaults.CLS_STABLE_MODEL_UPDATE_PROBABILITY,
    ) -> Dict[str, Component]:
        return {
            "memory_loss": WeightedCustomLossComponent(
                loss_fn=loss_fn, weight=alpha, sample_new_memory_batch=True
            ),
            "cls_loss": WeightedCLSLossComponent(
                weight=beta,
                sample_new_memory_batch=False,
                model=model,
                plastic_model_update_weight=plastic_model_update_weight,
                stable_model_update_weight=stable_model_update_weight,
                plastic_model_update_probability=plastic_model_update_probability,
                stable_model_update_probability=stable_model_update_probability,
            ),
        }


class SuperExperienceReplayLearner(BaseExperienceReplayLearner):
    """A learner that implements a selected combination of methods.

    Args:
        der_alpha: The weight of the mean squared error loss component between memorised logits and
            the current logits on the memory data.
        der_beta: The weight of the cross-entropy loss component between memorised targets and the
            current logits on the memory data.
        sp_shrink_factor: Shrinking value applied with respect to shrink and perturbation.
        sp_sigma: Standard deviation applied with respect to shrink and perturbation.
        cls_alpha: Scaling value for the consistency loss added to the base cross-entropy loss.
        cls_stable_model_update_weight: The starting weight for the exponential moving average to
            update the stable model copy.
        cls_plastic_model_update_weight: The starting weight for the exponential moving average to
            update the plastic model copy.
        cls_stable_model_update_probability: The probability to update the stable model copy.
        cls_plastic_model_update_probability: The probability to update the plastic model copy.
        pod_alpha: Scaling value which scales the loss with respect to all intermediate
            representations.
        pod_distillation_type: Which distillation type to apply with respect to the intermediate
            representation.
        pod_normalize: Whether to normalize both the current and cached features before computing
            the Frobenius norm.
        ema_memory_update_gamma: The gamma used for exponential moving average to update the meta
            data with respect to the logits and intermediate representation, if there is some.
    """

    def __init__(
        self,
        der_alpha: float = defaults.SER_DER_ALPHA,
        der_beta: float = defaults.SER_DER_BETA,
        sp_shrink_factor: float = defaults.SER_SP_SHRINK_FACTOR,
        sp_sigma: float = defaults.SER_SP_SIGMA,
        cls_alpha: float = defaults.SER_CLS_ALPHA,
        cls_stable_model_update_weight: float = defaults.SER_CLS_STABLE_MODEL_UPDATE_WEIGHT,
        cls_plastic_model_update_weight: float = defaults.SER_CLS_PLASTIC_MODEL_UPDATE_WEIGHT,
        cls_stable_model_update_probability: float = defaults.SER_CLS_STABLE_MODEL_UPDATE_PROBABILITY,  # noqa: E501
        cls_plastic_model_update_probability: float = defaults.SER_CLS_PLASTIC_MODEL_UPDATE_PROBABILITY,  # noqa: E501
        pod_alpha: float = defaults.SER_POD_ALPHA,
        pod_distillation_type: str = defaults.SER_POD_DISTILLATION_TYPE,
        pod_normalize: bool = defaults.SER_POD_NORMALIZE,
        ema_memory_update_gamma: float = defaults.EMA_MEMORY_UPDATE_GAMMA,
        **kwargs,
    ) -> None:
        components = self.components(
            model=kwargs["model"],
            loss_fn=kwargs["loss_fn"],
            der_alpha=der_alpha,
            der_beta=der_beta,
            sp_shrink_factor=sp_shrink_factor,
            sp_sigma=sp_sigma,
            cls_alpha=cls_alpha,
            cls_stable_model_update_weight=cls_stable_model_update_weight,
            cls_plastic_model_update_weight=cls_plastic_model_update_weight,
            cls_stable_model_update_probability=cls_stable_model_update_probability,
            cls_plastic_model_update_probability=cls_plastic_model_update_probability,
            pod_alpha=pod_alpha,
            pod_distillation_type=pod_distillation_type,
            pod_normalize=pod_normalize,
        )
        super().__init__(
            components=components,
            ema_memory_update_gamma=ema_memory_update_gamma,
            **kwargs,
        )

    def components(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        der_alpha: float = defaults.SER_DER_ALPHA,
        der_beta: float = defaults.SER_DER_BETA,
        sp_shrink_factor: float = defaults.SER_SP_SHRINK_FACTOR,
        sp_sigma: float = defaults.SER_SP_SIGMA,
        cls_alpha: float = defaults.SER_CLS_ALPHA,
        cls_stable_model_update_weight: float = defaults.SER_CLS_STABLE_MODEL_UPDATE_WEIGHT,
        cls_plastic_model_update_weight: float = defaults.SER_CLS_PLASTIC_MODEL_UPDATE_WEIGHT,
        cls_stable_model_update_probability: float = defaults.SER_CLS_STABLE_MODEL_UPDATE_PROBABILITY,  # noqa: E501
        cls_plastic_model_update_probability: float = defaults.SER_CLS_PLASTIC_MODEL_UPDATE_PROBABILITY,  # noqa: E501
        pod_alpha: float = defaults.SER_POD_ALPHA,
        pod_distillation_type: str = defaults.SER_POD_DISTILLATION_TYPE,
        pod_normalize: bool = defaults.SER_POD_NORMALIZE,
    ) -> Dict[str, Component]:
        return {
            "mse_loss": WeightedMeanSquaredErrorLossComponent(
                weight=der_alpha, sample_new_memory_batch=True
            ),
            "memory_loss": WeightedCustomLossComponent(
                loss_fn=loss_fn, weight=der_beta, sample_new_memory_batch=True
            ),
            "cls_loss": WeightedCLSLossComponent(
                weight=cls_alpha,
                sample_new_memory_batch=False,
                model=model,
                stable_model_update_weight=cls_stable_model_update_weight,
                plastic_model_update_weight=cls_plastic_model_update_weight,
                stable_model_update_probability=cls_stable_model_update_probability,
                plastic_model_update_probability=cls_plastic_model_update_probability,
            ),
            "shrink_perturb": ShrinkAndPerturbReinitializationComponent(
                shrink_factor=sp_shrink_factor, sigma=sp_sigma
            ),
            "pod_loss": WeightedPooledOutputDistillationLossComponent(
                weight=pod_alpha,
                sample_new_memory_batch=True,
                distillation_type=pod_distillation_type,
                normalize=pod_normalize,
            ),
        }


class ExperienceReplayModelUpdater(SingleTrainingLoopUpdater):
    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[Parameter]], Optimizer],
        memory_size: int,
        batch_memory_frac: int = defaults.BATCH_MEMORY_FRAC,
        loss_weight: float = defaults.LOSS_WEIGHT,
        ema_memory_update_gamma: float = defaults.EMA_MEMORY_UPDATE_GAMMA,
        loss_normalization: int = defaults.LOSS_NORMALIZATION,
        alpha: float = defaults.ER_ALPHA,
        learning_rate_scheduler: Optional[partial] = None,
        learning_rate_scheduler_interval: defaults.SUPPORTED_LR_SCHEDULER_INTERVAL_TYPE = defaults.LR_SCHEDULER_INTERVAL,  # noqa: E501
        batch_size: int = defaults.BATCH_SIZE,
        input_state_folder: Optional[str] = None,
        output_state_folder: Optional[str] = None,
        max_epochs: int = defaults.MAX_EPOCHS,
        train_transform: Optional[Callable] = None,
        train_target_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        test_target_transform: Optional[Callable] = None,
        buffer_transform: Optional[Callable] = None,
        buffer_target_transform: Optional[Callable] = None,
        metric: Optional[str] = None,
        mode: defaults.SUPPORTED_TUNING_MODE_TYPE = "min",
        logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        early_stopping_enabled: bool = False,
        logger: Logger = defaults.LOGGER(**defaults.LOGGER_KWARGS),
        accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
        devices: Optional[int] = None,
        strategy: str = defaults.DISTRIBUTED_STRATEGY,
        precision: str = defaults.PRECISION,
        seed: int = defaults.SEED,
        deterministic_trainer: bool = defaults.DETERMINISTIC_TRAINER,
        gradient_clip_val: Optional[float] = defaults.GRADIENT_CLIP_VAL,
        gradient_clip_algorithm: Optional[str] = defaults.GRADIENT_CLIP_ALGORITHM,
        mask_unused_classes: bool = defaults.MASK_UNUSED_CLASSES,
    ):
        learner_kwargs = {
            "memory_size": memory_size,
            "batch_memory_frac": batch_memory_frac,
            "loss_weight": loss_weight,
            "ema_memory_update_gamma": ema_memory_update_gamma,
            "loss_normalization": loss_normalization,
            "alpha": alpha,
            "batch_size": batch_size,
            "seed": seed,
        }
        super().__init__(
            model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=ExperienceReplayLearner,
            learner_kwargs=learner_kwargs,
            input_state_folder=input_state_folder,
            output_state_folder=output_state_folder,
            max_epochs=max_epochs,
            learning_rate_scheduler=learning_rate_scheduler,
            learning_rate_scheduler_interval=learning_rate_scheduler_interval,
            train_transform=train_transform,
            train_target_transform=train_target_transform,
            test_transform=test_transform,
            test_target_transform=test_target_transform,
            buffer_transform=buffer_transform,
            buffer_target_transform=buffer_target_transform,
            metric=metric,
            mode=mode,
            logged_metrics=logged_metrics,
            early_stopping_enabled=early_stopping_enabled,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            deterministic_trainer=deterministic_trainer,
            gradient_clip_algorithm=gradient_clip_algorithm,
            gradient_clip_val=gradient_clip_val,
            mask_unused_classes=mask_unused_classes,
        )


class DarkExperienceReplayModelUpdater(SingleTrainingLoopUpdater):
    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[Parameter]], Optimizer],
        memory_size: int,
        batch_memory_frac: int = defaults.BATCH_MEMORY_FRAC,
        loss_weight: float = defaults.LOSS_WEIGHT,
        ema_memory_update_gamma: float = defaults.EMA_MEMORY_UPDATE_GAMMA,
        loss_normalization: int = defaults.LOSS_NORMALIZATION,
        alpha: float = defaults.DER_ALPHA,
        beta: float = defaults.DER_BETA,
        learning_rate_scheduler: Optional[partial] = None,
        learning_rate_scheduler_interval: defaults.SUPPORTED_LR_SCHEDULER_INTERVAL_TYPE = defaults.LR_SCHEDULER_INTERVAL,  # noqa: E501
        batch_size: int = defaults.BATCH_SIZE,
        input_state_folder: Optional[str] = None,
        output_state_folder: Optional[str] = None,
        max_epochs: int = defaults.MAX_EPOCHS,
        train_transform: Optional[Callable] = None,
        train_target_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        test_target_transform: Optional[Callable] = None,
        buffer_transform: Optional[Callable] = None,
        buffer_target_transform: Optional[Callable] = None,
        metric: Optional[str] = None,
        mode: defaults.SUPPORTED_TUNING_MODE_TYPE = "min",
        logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        early_stopping_enabled: bool = False,
        logger: Logger = defaults.LOGGER(**defaults.LOGGER_KWARGS),
        accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
        devices: Optional[int] = None,
        strategy: str = defaults.DISTRIBUTED_STRATEGY,
        precision: str = defaults.PRECISION,
        seed: int = defaults.SEED,
        deterministic_trainer: bool = defaults.DETERMINISTIC_TRAINER,
        gradient_clip_val: Optional[float] = defaults.GRADIENT_CLIP_VAL,
        gradient_clip_algorithm: Optional[str] = defaults.GRADIENT_CLIP_ALGORITHM,
        mask_unused_classes: bool = defaults.MASK_UNUSED_CLASSES,
    ):
        learner_kwargs = {
            "memory_size": memory_size,
            "batch_memory_frac": batch_memory_frac,
            "loss_weight": loss_weight,
            "ema_memory_update_gamma": ema_memory_update_gamma,
            "loss_normalization": loss_normalization,
            "alpha": alpha,
            "beta": beta,
            "batch_size": batch_size,
            "seed": seed,
        }
        super().__init__(
            model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=DarkExperienceReplayLearner,
            learner_kwargs=learner_kwargs,
            input_state_folder=input_state_folder,
            output_state_folder=output_state_folder,
            max_epochs=max_epochs,
            learning_rate_scheduler=learning_rate_scheduler,
            learning_rate_scheduler_interval=learning_rate_scheduler_interval,
            train_transform=train_transform,
            train_target_transform=train_target_transform,
            test_transform=test_transform,
            test_target_transform=test_target_transform,
            buffer_transform=buffer_transform,
            buffer_target_transform=buffer_target_transform,
            metric=metric,
            mode=mode,
            logged_metrics=logged_metrics,
            early_stopping_enabled=early_stopping_enabled,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            deterministic_trainer=deterministic_trainer,
            gradient_clip_algorithm=gradient_clip_algorithm,
            gradient_clip_val=gradient_clip_val,
            mask_unused_classes=mask_unused_classes,
        )


class PooledOutputDistillationExperienceReplayModelUpdater(SingleTrainingLoopUpdater):
    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[Parameter]], Optimizer],
        memory_size: int,
        batch_memory_frac: int = defaults.BATCH_MEMORY_FRAC,
        loss_weight: float = defaults.LOSS_WEIGHT,
        ema_memory_update_gamma: float = defaults.EMA_MEMORY_UPDATE_GAMMA,
        loss_normalization: int = defaults.LOSS_NORMALIZATION,
        alpha: float = defaults.POD_ALPHA,
        distillation_type: str = defaults.POD_DISTILLATION_TYPE,
        normalize: bool = defaults.POD_NORMALIZE,
        learning_rate_scheduler: Optional[partial] = None,
        learning_rate_scheduler_interval: defaults.SUPPORTED_LR_SCHEDULER_INTERVAL_TYPE = defaults.LR_SCHEDULER_INTERVAL,  # noqa: E501
        batch_size: int = defaults.BATCH_SIZE,
        input_state_folder: Optional[str] = None,
        output_state_folder: Optional[str] = None,
        max_epochs: int = defaults.MAX_EPOCHS,
        train_transform: Optional[Callable] = None,
        train_target_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        test_target_transform: Optional[Callable] = None,
        buffer_transform: Optional[Callable] = None,
        buffer_target_transform: Optional[Callable] = None,
        metric: Optional[str] = None,
        mode: defaults.SUPPORTED_TUNING_MODE_TYPE = "min",
        logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        early_stopping_enabled: bool = False,
        logger: Logger = defaults.LOGGER(**defaults.LOGGER_KWARGS),
        accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
        devices: Optional[int] = None,
        strategy: str = defaults.DISTRIBUTED_STRATEGY,
        precision: str = defaults.PRECISION,
        seed: int = defaults.SEED,
        deterministic_trainer: bool = defaults.DETERMINISTIC_TRAINER,
        gradient_clip_val: Optional[float] = defaults.GRADIENT_CLIP_VAL,
        gradient_clip_algorithm: Optional[str] = defaults.GRADIENT_CLIP_ALGORITHM,
        mask_unused_classes: bool = defaults.MASK_UNUSED_CLASSES,
    ):
        learner_kwargs = {
            "memory_size": memory_size,
            "batch_memory_frac": batch_memory_frac,
            "loss_weight": loss_weight,
            "ema_memory_update_gamma": ema_memory_update_gamma,
            "loss_normalization": loss_normalization,
            "alpha": alpha,
            "distillation_type": distillation_type,
            "normalize": normalize,
            "batch_size": batch_size,
            "seed": seed,
        }
        super().__init__(
            model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=PooledOutputDistillationExperienceReplayLearner,
            learner_kwargs=learner_kwargs,
            input_state_folder=input_state_folder,
            output_state_folder=output_state_folder,
            max_epochs=max_epochs,
            learning_rate_scheduler=learning_rate_scheduler,
            learning_rate_scheduler_interval=learning_rate_scheduler_interval,
            train_transform=train_transform,
            train_target_transform=train_target_transform,
            test_transform=test_transform,
            test_target_transform=test_target_transform,
            buffer_transform=buffer_transform,
            buffer_target_transform=buffer_target_transform,
            metric=metric,
            mode=mode,
            logged_metrics=logged_metrics,
            early_stopping_enabled=early_stopping_enabled,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            deterministic_trainer=deterministic_trainer,
            gradient_clip_algorithm=gradient_clip_algorithm,
            gradient_clip_val=gradient_clip_val,
            mask_unused_classes=mask_unused_classes,
        )


class CLSExperienceReplayModelUpdater(SingleTrainingLoopUpdater):
    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[Parameter]], Optimizer],
        memory_size: int,
        batch_memory_frac: int = defaults.BATCH_MEMORY_FRAC,
        loss_weight: float = defaults.LOSS_WEIGHT,
        ema_memory_update_gamma: float = defaults.EMA_MEMORY_UPDATE_GAMMA,
        loss_normalization: int = defaults.LOSS_NORMALIZATION,
        alpha: float = defaults.CLS_ALPHA,
        beta: float = defaults.CLS_BETA,
        stable_model_update_weight: float = defaults.CLS_STABLE_MODEL_UPDATE_WEIGHT,
        plastic_model_update_weight: float = defaults.CLS_PLASTIC_MODEL_UPDATE_WEIGHT,
        stable_model_update_probability: float = defaults.CLS_STABLE_MODEL_UPDATE_PROBABILITY,
        plastic_model_update_probability: float = defaults.CLS_PLASTIC_MODEL_UPDATE_PROBABILITY,
        learning_rate_scheduler: Optional[partial] = None,
        learning_rate_scheduler_interval: defaults.SUPPORTED_LR_SCHEDULER_INTERVAL_TYPE = defaults.LR_SCHEDULER_INTERVAL,  # noqa: E501
        batch_size: int = defaults.BATCH_SIZE,
        input_state_folder: Optional[str] = None,
        output_state_folder: Optional[str] = None,
        max_epochs: int = defaults.MAX_EPOCHS,
        train_transform: Optional[Callable] = None,
        train_target_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        test_target_transform: Optional[Callable] = None,
        buffer_transform: Optional[Callable] = None,
        buffer_target_transform: Optional[Callable] = None,
        metric: Optional[str] = None,
        mode: defaults.SUPPORTED_TUNING_MODE_TYPE = "min",
        logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        early_stopping_enabled: bool = False,
        logger: Logger = defaults.LOGGER(**defaults.LOGGER_KWARGS),
        accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
        devices: Optional[int] = None,
        strategy: str = defaults.DISTRIBUTED_STRATEGY,
        precision: str = defaults.PRECISION,
        seed: int = defaults.SEED,
        deterministic_trainer: bool = defaults.DETERMINISTIC_TRAINER,
        gradient_clip_val: Optional[float] = defaults.GRADIENT_CLIP_VAL,
        gradient_clip_algorithm: Optional[str] = defaults.GRADIENT_CLIP_ALGORITHM,
        mask_unused_classes: bool = defaults.MASK_UNUSED_CLASSES,
    ):
        learner_kwargs = {
            "memory_size": memory_size,
            "batch_memory_frac": batch_memory_frac,
            "loss_weight": loss_weight,
            "ema_memory_update_gamma": ema_memory_update_gamma,
            "loss_normalization": loss_normalization,
            "alpha": alpha,
            "beta": beta,
            "stable_model_update_weight": stable_model_update_weight,
            "plastic_model_update_weight": plastic_model_update_weight,
            "stable_model_update_probability": stable_model_update_probability,
            "plastic_model_update_probability": plastic_model_update_probability,
            "batch_size": batch_size,
            "seed": seed,
        }
        super().__init__(
            model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=CLSExperienceReplayLearner,
            learner_kwargs=learner_kwargs,
            input_state_folder=input_state_folder,
            output_state_folder=output_state_folder,
            max_epochs=max_epochs,
            learning_rate_scheduler=learning_rate_scheduler,
            learning_rate_scheduler_interval=learning_rate_scheduler_interval,
            train_transform=train_transform,
            train_target_transform=train_target_transform,
            test_transform=test_transform,
            test_target_transform=test_target_transform,
            buffer_transform=buffer_transform,
            buffer_target_transform=buffer_target_transform,
            metric=metric,
            mode=mode,
            logged_metrics=logged_metrics,
            early_stopping_enabled=early_stopping_enabled,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            deterministic_trainer=deterministic_trainer,
            gradient_clip_algorithm=gradient_clip_algorithm,
            gradient_clip_val=gradient_clip_val,
            mask_unused_classes=mask_unused_classes,
        )


class SuperExperienceReplayModelUpdater(SingleTrainingLoopUpdater):
    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[Parameter]], Optimizer],
        memory_size: int,
        batch_memory_frac: int = defaults.BATCH_MEMORY_FRAC,
        loss_weight: float = defaults.LOSS_WEIGHT,
        ema_memory_update_gamma: float = defaults.EMA_MEMORY_UPDATE_GAMMA,
        loss_normalization: int = defaults.LOSS_NORMALIZATION,
        der_alpha: float = defaults.SER_DER_ALPHA,
        der_beta: float = defaults.SER_DER_BETA,
        sp_shrink_factor: float = defaults.SER_SP_SHRINK_FACTOR,
        sp_sigma: float = defaults.SER_SP_SIGMA,
        cls_alpha: float = defaults.SER_CLS_ALPHA,
        cls_stable_model_update_weight: float = defaults.SER_CLS_STABLE_MODEL_UPDATE_WEIGHT,
        cls_plastic_model_update_weight: float = defaults.SER_CLS_PLASTIC_MODEL_UPDATE_WEIGHT,
        cls_stable_model_update_probability: float = defaults.SER_CLS_STABLE_MODEL_UPDATE_PROBABILITY,  # noqa: E501
        cls_plastic_model_update_probability: float = defaults.SER_CLS_PLASTIC_MODEL_UPDATE_PROBABILITY,  # noqa: E501
        pod_alpha: float = defaults.SER_POD_ALPHA,
        pod_distillation_type: str = defaults.SER_POD_DISTILLATION_TYPE,
        pod_normalize: bool = defaults.SER_POD_NORMALIZE,
        learning_rate_scheduler: Optional[partial] = None,
        learning_rate_scheduler_interval: defaults.SUPPORTED_LR_SCHEDULER_INTERVAL_TYPE = defaults.LR_SCHEDULER_INTERVAL,  # noqa: E501
        batch_size: int = defaults.BATCH_SIZE,
        input_state_folder: Optional[str] = None,
        output_state_folder: Optional[str] = None,
        max_epochs: int = defaults.MAX_EPOCHS,
        train_transform: Optional[Callable] = None,
        train_target_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        test_target_transform: Optional[Callable] = None,
        buffer_transform: Optional[Callable] = None,
        buffer_target_transform: Optional[Callable] = None,
        metric: Optional[str] = None,
        mode: defaults.SUPPORTED_TUNING_MODE_TYPE = "min",
        logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        early_stopping_enabled: bool = False,
        logger: Logger = defaults.LOGGER(**defaults.LOGGER_KWARGS),
        accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
        devices: Optional[int] = None,
        strategy: str = defaults.DISTRIBUTED_STRATEGY,
        precision: str = defaults.PRECISION,
        seed: int = defaults.SEED,
        deterministic_trainer: bool = defaults.DETERMINISTIC_TRAINER,
        gradient_clip_val: Optional[float] = defaults.GRADIENT_CLIP_VAL,
        gradient_clip_algorithm: Optional[str] = defaults.GRADIENT_CLIP_ALGORITHM,
        mask_unused_classes: bool = defaults.MASK_UNUSED_CLASSES,
    ):
        learner_kwargs = {
            "memory_size": memory_size,
            "batch_memory_frac": batch_memory_frac,
            "loss_weight": loss_weight,
            "ema_memory_update_gamma": ema_memory_update_gamma,
            "loss_normalization": loss_normalization,
            "der_alpha": der_alpha,
            "der_beta": der_beta,
            "sp_shrink_factor": sp_shrink_factor,
            "sp_sigma": sp_sigma,
            "cls_alpha": cls_alpha,
            "cls_stable_model_update_weight": cls_stable_model_update_weight,
            "cls_plastic_model_update_weight": cls_plastic_model_update_weight,
            "cls_stable_model_update_probability": cls_stable_model_update_probability,
            "cls_plastic_model_update_probability": cls_plastic_model_update_probability,
            "pod_alpha": pod_alpha,
            "pod_distillation_type": pod_distillation_type,
            "pod_normalize": pod_normalize,
            "batch_size": batch_size,
            "seed": seed,
        }
        super().__init__(
            model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=SuperExperienceReplayLearner,
            learner_kwargs=learner_kwargs,
            input_state_folder=input_state_folder,
            output_state_folder=output_state_folder,
            max_epochs=max_epochs,
            learning_rate_scheduler=learning_rate_scheduler,
            learning_rate_scheduler_interval=learning_rate_scheduler_interval,
            train_transform=train_transform,
            train_target_transform=train_target_transform,
            test_transform=test_transform,
            test_target_transform=test_target_transform,
            buffer_transform=buffer_transform,
            buffer_target_transform=buffer_target_transform,
            metric=metric,
            mode=mode,
            logged_metrics=logged_metrics,
            early_stopping_enabled=early_stopping_enabled,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            deterministic_trainer=deterministic_trainer,
            gradient_clip_algorithm=gradient_clip_algorithm,
            gradient_clip_val=gradient_clip_val,
            mask_unused_classes=mask_unused_classes,
        )
