import torch
from functools import partial

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from deep_vit import Transformer

# submodule oluştur
check_fn = lambda submodule: isinstance(submodule, Transformer)

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    ofload_to_cpu=False,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

model = FSDP(
    model,
    auto_wrap_policy=wraping_policy,
    mixed_precision=mp_policy,
    sharding_strategy=model_sharding_strategy,
    device_id=torch.cuda.current_device(),
)

apply_activation_checkpointing(
    model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
)