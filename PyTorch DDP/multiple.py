import torch
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler # girdi verilerini alarak tüm gpu lara vermeye yarar
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

# word size bir gruptaki toplam süreç sayısıdır


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Her sürecin benzersiz tanımlayıcısı 0 ile -1 alır
        world_size: Gruptaki toplam süreç sayısı
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    
    

class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,
        ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = torch.nn.CrossEntropyLoss()(output, targets)
        loss.backward()
        self.optimizer.step()


    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU:{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict() # DDP de model.module.state_dict
        #torch.save(ckp, "checkpoint.pt") kaggle da dosya yazmaya izin vermiyor bu sebeple gerek yok
        print(f"Epoch {epoch} | Training checkpoint save at checkpoint.pt")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0: # değiştirildi gpu_id 0 olduğunda
                self._save_checkpoint(epoch)

def load_train_objs():
    train_set = MyTrainDataset(2048)
    model = torch.nn.Linear(20, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer
    
def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size = batch_size,
        pin_memory=True,
        shuffle=False, # shuffle False yapılır
        sampler = DistributedSampler(dataset) # DistributedSampler ile her gpu ya verileri dağıtmaya yarar
    )
    
def main(rank: int, world_size: int, total_epochs: int, save_every: int):
    ddp_setup(rank, world_size) # ddp yi başlatıyoruz
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset,batch_size=32)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group() # eğitim bittiğinde tüm process guruplarını temizler


if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    world_size = torch.cuda.device_count()
    # ddp de ana fonksiyonu mp.spawn ile çalıştıracağız argümanları ve world_size belirlenecek
    mp.spawn(main, args=(world_size, total_epochs, save_every), nprocs=world_size)
    