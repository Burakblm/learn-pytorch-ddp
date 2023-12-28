from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
import torch

# Mevcut üç parçalama stratejisi - tradeoff bellek boyutu ve iletişim yükü:
ShardingStrategy.FULL_SHARD # default! Model, optimize edici ve degradenin tümü parçalanmıştır (iletişim halindedir) ... maksimum model boyutu desteği
ShardingStrategy.SHARD_GRAD_OP # Zero mode - model parametreleri ileri geçişten sonra serbest bırakılmaz, bu da iletişim ihtiyaçlarını azaltır
ShardingStrategy.NO_SHARD # DDP mode - her GPU modelin, optimize edicinin ve degradelerin tam bir kopyasını saklar

# yalnızca grad senkronizasyonu gerekli

# Future support:
ShardingStrategy.HYBRID_SHARD #FSDP Her düğümde tam parça var, ancak her düğüm arasında Parça Yok (DDP).

model = FSDP(
    model,
    auto_wrap_policy=my_auto_wrap_policy,
    mixed_precision=mp_policy,
    backward_prefetch=prefetch_policy,
    #sharing control
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,

    device_id=torch.cuda.current_device(),
    forward_prefetch=True,
)

"""
gpu ya sığan model boyutları

DDP 750 M
Zero2 2 B
FSDP Full 2.5 B

"""