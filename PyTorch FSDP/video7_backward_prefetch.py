import torch

from torch.distributed.fsdp import(
    FullyShardedDataParallel as FSDP,
    BackwardPrefetch,
    StateDictType,
    FullStateDictConfig,
    LocalStateDictConfig,
    ShardingStrategy,
)

prefetch_policy = BackwardPrefetch.BACKWARD_PRE # BackwardPreetch.BACKWARD_POST or None

model = FSDP(
    model,
    auto_wrap_policy=my_auto_warp_policy,
    mixed_precision=mp_policy,
    backward_prefetch=prefetch_policy,
    device_id=torch.cuda.current_device(),
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    forward_prefetch=True,
)

