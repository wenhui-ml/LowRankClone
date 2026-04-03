from datasets import IterableDataset
from transformers import TrainingArguments, Trainer
import torch
from torch import nn
import os

os.environ["LOCAL_RANK"] = "1"
os.environ["WORLD_SIZE"] = "2"
os.environ["RANK"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
torch.distributed.init_process_group(backend="nccl")

def gen():
    for i in range(10):
        yield {"x": [i], "labels": i}

dataset = IterableDataset.from_generator(gen)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(1, 1)
    def forward(self, x, labels=None):
        loss = torch.tensor(0.0, requires_grad=True)
        return {"loss": loss, "logits": x}

model = DummyModel()
args = TrainingArguments(output_dir="./test_out", max_steps=2, dataloader_num_workers=0)
trainer = Trainer(model=model, args=args, train_dataset=dataset)
loader = trainer.get_train_dataloader()
for b in loader:
    print("RANK 1 yields:", b)
    break
