#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_smolvla import SmolVLAConfig


@ProcessorStepRegistry.register("mask_state_processor")
@dataclass
class MaskStateProcessorStep(ProcessorStep):
    """Zeroes out specified dimensions of observation.state.

    Used when training a model that should not see certain state dimensions,
    while earlier steps (e.g., RelativeActionsProcessorStep) still use the real state.
    Should be placed AFTER NormalizerProcessorStep so that masked dims are exact zeros.

    Attributes:
        enabled: Whether to apply masking.
        mask_names: State feature names to zero out.
        state_names: All state feature names (from dataset metadata).
    """

    enabled: bool = False
    mask_names: list[str] = field(default_factory=list)
    state_names: list[str] | None = field(default=None)

    def _build_mask(self, state_dim: int) -> list[bool]:
        """Returns list where True = zero this dim."""
        if not self.mask_names or self.state_names is None:
            return [False] * state_dim
        mask_tokens = {n.lower() for n in self.mask_names}
        result = []
        for name in self.state_names[:state_dim]:
            result.append(name.lower() in mask_tokens)
        result.extend([False] * (state_dim - len(result)))
        return result

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE)
        if state is None:
            return new_transition

        mask = self._build_mask(state.shape[-1])
        mask_t = torch.tensor(mask, dtype=torch.bool, device=state.device)
        new_state = state.clone()
        new_state[..., mask_t] = 0.0
        observation[OBS_STATE] = new_state
        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "mask_names": self.mask_names, "state_names": self.state_names}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Adding a batch dimension.
    3.  Converting absolute actions to relative (if enabled).
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.
    7.  Normalizing input and output features based on dataset statistics.
    8.  Masking specified state dimensions (if enabled).

    The post-processing pipeline handles the model's output by:
    1.  Unnormalizing the output actions to their original scale.
    2.  Converting relative actions back to absolute (if enabled).
    3.  Moving data to the CPU.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_features", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    has_state_mask = bool(config.state_mask)

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        relative_step,
        NewLineTaskProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        MaskStateProcessorStep(
            enabled=has_state_mask,
            mask_names=config.state_mask,
            state_names=getattr(config, "state_feature_names", None),
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
