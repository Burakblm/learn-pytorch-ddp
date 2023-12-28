import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    LocalStateDictConfig,
)

from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)

from pathlib import Path



def save_distrubuted_model_checkpoint(model, rank, cfg, epoch=1):

    if rank == 0:
        print(f"Starting distrubuted checkpoint save...")

    if cfg.checkpoint_type == StateDictType.LOCAL_STATE_DICT:
        # create writer to current path
        folder_name = cfg.dist_checkpoint_root_file + "/" + cfg.dist_checkpoint_folder + "-" + cfg.model_name
        save_dir = Path.cwd() / folder_name

        writer = FileSystemWriter(save_dir)

        with FSDP.state_dict_type(
            model,
            StateDictType.LOCAL_STATE_DICT
        ):
            state_dict = model.state_dict()

        # write out distrubutuedn checkpoint
        save_state_dict(state_dict, writer)

        if rank == 0:
            print(f"distrubuted checkpoint saved at {save_dir}")


def load_distrubuted_model_checkpoint(model, rank, cfg):
    if cfg.checkpoint_type == StateDictType.LOCAL_STATE_DICT:
        print(f"loading distrubuted checkpoint rank {rank}...")
        folder_name = cfg.dist_checkpoint_root_folder + "/" + cfg.dist_checkpoint_folder + "-" + cfg.model_name

        checkdir = Path.cwd() / folder_name

        if not checkdir.exists():
            if rank == 0:
                print(f"no Checkpoint directory found ... skiping")
            return
        
        reader = FileSystemReader(checkdir)

        with FSDP.state_dict_type(
            model,
            StateDictType.LOCAL_STATE_DICT,
        ):
            state_dict = model.state_dict()
            load_state_dict(state_dict, reader)
            model.load_state_dict(state_dict)

        print(f"local state loaded on rank {rank}")

# Genel Notlar:
# Daha büyük modeller için (20B+ CPU belleğinin sorun olabileceğini düşünün ve dağıtılmış kontrol noktalarının olduğu yer burasıdır)
# veya local_state_checkpoints parlıyor.
# dağıtılmış kontrol noktası tasarrufu, ağ ve bilgi işlem tarafında daha hızlıdır, ancak binlerce veriyi yazarken gecikme yaşanabilir
# dosya.
# Önemli - dağıtılmış bir kontrol noktasındaki tek tanımlayıcı meta veri dosyasıdır. Ancak kaydederseniz
# aynı dizine başka bir kontrol noktası daha yazılırsa, meta veri dosyasının üzerine yazılacak ve önceki kontrol noktası kaybedilecek
# (ancak ilgilenmeniz gereken çok sayıda ek dosyanız olacak).

