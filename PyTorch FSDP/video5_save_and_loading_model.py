import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)

model = build_model(cfg.model_name)

# laod checkpoint for model
if(
    cfg.load_model_checkpoint
    and cfg.checkpoint_type == StateDictType.FULL_STATE_DICT
):
    model.checkpointing.laod_model_checkpoint(model, rank, cfg)


def load_model_checkpoint(model, rank, cfg, verbose=True):
    if rank != 0:
        return
    
    full_state_dict_model_path = (
        Path.cwd() / cfg.checkpoint_folder / cfg.checkpoint_model_filename
    )

    # is it present
    if not full_state_dict_model_path.is_file():
        print(f"model checkpoint {full_state_dict_model_path} not present Returning..."
        )
        return
    
    # load the checkpoint
    model_checkpoint = torch.load(full_state_dict_model_path)
    # integrate into loaded model
    model.load_state_dict(model_checkpoint)

    if cfg.verbose:
        print(f"model checkpoint loaded to rank0 cpu")


model = FSDP(
    model,
    auto_wrap_policy=my_auto_wrap_policy,
    mixed_precision=mp_policy,
    #backward_prefetch=prefetch_policy
    device_id=torch.cuda.current_device(),
    sharding_strategy=ShardingStrategy.FULL_SHARD, #Zero2
    #cpu_offload=cpu_policy
    forward_prefetch=True,
)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=8e-4, weight_decay=0.005
)

if cfg.load_optimizer:
    model.checkpointing.load_optimizer_checkpoint(model, optimizer, rank, cfg)


def load_optimizer_checkpoint(model, optimizer, rank, cfg):
    opt_file_path = Path.cwd() / cfg.checkpoint_folder / cfg.optimizer_checkpoint_file

    if not opt_file_path.is_file():
        print(
            f"warning - optimizer checkpoint not present {opt_file_path} Returning..."
        )
    
    full_osd = None

    if rank == 0:
        full_osd = torch.load(opt_file_path)

        if cfg.verbose:
            print(f"loaded full osd on rank 0")
    
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)

    if cfg.verbose:
        print(f"optimizer shard loaded on rank {rank}")


# training loop



# Tekrar tekrar yapmaktan kaçınmak için tekil kaydetme politikaları oluşturun
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

def save_model_checkpoint(
    model,
    optimizer,
    rank,
    cfg,
    epoch=1,
):
    """saving model via rank0 cpu stramming and full_state_dict"""

    # saving with rank0 cpu
    if not cfg.checkpoint_type == StateDictType.FULL_STATE_DICT:
        print(f"unable to handle checkpoint type {cfg.checkpoint_type}, aborting")

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()

    if cfg.verbose:
        print(f"saving proccess rank: {rank} done w model state dict")

    if rank == 0:
        print(f"saving model...")
        #create save path
        save_dir = Path.cwd() / cfg.checkpoint_folder
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = cfg.model_save_name + "-" + str(epoch) + ".pt"
        save_full_path = str(save_dir) + "/" + save_name

        # save model
        torch.save(cpu_state, save_full_path)

        if cfg.verbose:
            print(f"model checkpoint saved for epoch {epoch} at {save_full_path}")


def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """save optimizer state via full state dict"""

    if cfg.verbose:
        print(f"--> optim state call on rank {rank}")

    # pull all sharded optimizer states to rank0 cpu..
    
    optim_state = FSDP.full_optim_state_dict(model, optimizer)
    
    if cfg.verbose:
        print(f"optim state dict ready on {rank} and len of {len(optim_state)}")

    if rank == 0:
        save_dir = Path.cwd() / cfg.checkpoint_folder
        save_dir.mkdir(parents=True, exist_ok=True)

        opt_save_name = (
            cfg.optimizer_name + "-" + cfg.model_save_name + "-" + str(epoch) + ".pt"
        )

        opt_save_full_path = save_dir / opt_save_name

        print(f"saving optimizer state...")

        torch.save(optim_state, opt_save_full_path)

        print(f"saved {opt_save_full_path} to disk")