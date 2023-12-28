import functools
from dataclasses import dataclass

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from torch.distributed.fsdp import (
    ShardingStrategy,
    BackwardPrefetch,
)

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

@dataclass
class base_config:
    # seed
    seed: int = 2022
    verbose: bool = True # very slow...
    # how many mini batches to time with
    total_steps_to_run: int = 15
    num_epochs: int = 1

    # sharding policy
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    print_sharding_plan: bool = False

    run_profiler: bool = False

    # backward prefetch
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE

    # disable forward_prefetch since it currently doesn't work with action
    # checkpointing several cases
    forward_prefetch = False

    log: int = 1

    # dataloaders
    num_worker_dataloader: int = 2

    # policies
    use_mixed_precision: bool = False
    # this is only for fp32 scenerio...
    use_tf32: bool = False

    # activation checkpointing
    fsdp_activation_checkpointing: bool = True

    # validation
    run_validation: bool = False
    val_batch_size = 4

    # logging
    track_memory = True
    memory_report: bool = True
    nccl_debug_handler: bool = True
    distrubuted_debug: bool = True


def get_policy_base(blocks):
    recursive_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=blocks
    )
    return recursive_policy

def fsdp_checkpointing_base(model, blocks):
    """
    modele etkinleştirme kontrol noktasi uygulama, 
    model güncellendiğinden None değerini döndürür
    """
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, blocks)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=check_fn
    )