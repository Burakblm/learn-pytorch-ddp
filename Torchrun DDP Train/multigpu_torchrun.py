import torch
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler # girdi verilerini alarak tüm gpu lara vermeye yarar
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

# word size bir gruptaki toplam süreç sayısıdır


def ddp_setup(): # Torchrun bunu bizim için ayarlayacak
    init_process_group(backend="nccl")
    
    
    

class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            save_every: int,
            snapshot_path: str,
        ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        if os.path.exists(snapshot_path): # eğer snapshhot path da bir dosya varsa yükleme yapar
            print("Loading Snapshot")
            self._load_snapshot(snapshot_path)
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

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

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()  #DDP de model.module.state_dict
        snapshot["EPOCHS_RUN"] = epoch
        #torch.save(snapshot, "snapshot.pt") #kaggle da dosya yazmaya izin vermiyor bu sebeple gerek yok
        print(f"Epoch {epoch} | Training snapshot save at snapshot.pt")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs): # her defasında sıfırdan başlamasına gerek yok bu yüzden kalınan epoch dan devam eder
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0: # değiştirildi gpu_id 0 olduğunda
                self._save_snapshot(epoch)

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
    
def main(total_epochs: int, save_every: int, snapshot_path: str = "snapshot.pt"):
    ddp_setup() # ddp yi başlatıyoruz
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset,batch_size=32)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group() # eğitim bittiğinde tüm process guruplarını temizler


if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    main(save_every, total_epochs)