import torch
from torch.distributions import Bernoulli


torch.manual_seed(2022)
torch.cuda.manual_seed(2022)

grad = torch.randn(3,3)
reserve_p = .30


r = grad.new_full(size=grad.size(), fill_value=reserve_p)

rdist = Bernoulli(r)

rp = rdist.sample()

amplifier = rp / reserve_p

newgrad = grad * amplifier


from ChildTuningOptimizer import ChildTuningAdamW

model = torch.nn.Linear(100, 200)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
)

model = FSDP(
    model,
    auto_wrap_policy=my_auto_wrap_policy,
    mixed_precision=mp_policy,
    backward_prefetch=prefetch_policy,
    sharding_strategy=cfg.sharding_strategy,
    device_id=torch.cuda.current_device(),
    forward_prefetch=cfg.forward_prefetch,
)

optimizer = ChildTuningAdamW(model.parameters(), lr=4e-8, reserve_p=0.35, mode="taskfree")
