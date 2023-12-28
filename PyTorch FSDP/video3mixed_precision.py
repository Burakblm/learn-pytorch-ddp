import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)

bfloatPolicy = MixedPrecision(
    # param precision
    param_dtype=torch.bfloat16,
    # Gradient communication precision
    reduce_dtype=torch.bfloat16,
    # buffer precision
    buffer_dtype=torch.bfloat16,
)

comboPolicy = MixedPrecision(
    # param precision
    param_dtype=torch.bfloat16,
    # Gradient communication precision
    reduce_dtype=torch.float32,
    # buffer precision
    buffer_dtype=torch.float32,
)

model = FSDP(
    model,
    auto_wrap_policy=my_auto_wrap_policy,

    mixed_precision=bfloatPolicy, # <- mixed precision policy

    backward_prefetch=prefetch_policy,
    sharding_strategy=cfg.sharding_strategy,
    device_id=torch.cuda.current_device(),
    forward_prefetch=True,
)

# bflaot16 desteklenen cihazla için 

from pkg_resources import packaging
import torch.cuda.nccl as nccl
import torch.distributed as dist

verify_bfloat_supported = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported(),
)

basic_bfloat_ready = torch.cuda.is_bf16_supported()

if cfg.use_fp16:
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    scaler = ShardedGradScaler()

loss = output["loss"]
if scaler:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()



torch.backends.cuda.matmul.allow_tf32 = True