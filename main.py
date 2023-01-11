import argparse
import sys

import torch

from data import CorruptMnist
from model import MyAwesomeModel

import matplotlib.pyplot as plt
import wandb




device = "cuda" if torch.cuda.is_available() else "cpu"
            
    
print("Training day and night")
parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--lr', default=1e-3)
# add any additional argument that you want
args = parser.parse_args(sys.argv[2:])
print(args)

wandb.init(config=args)
        
# TODO: Implement training loop here
model = MyAwesomeModel()
model = model.to(device)
wandb.watch(model)
train_set = CorruptMnist(train=True)
dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()
        
n_epoch = 5
for epoch in range(n_epoch):
    loss_tracker = []
    for batch in dataloader:
        optimizer.zero_grad()
        x, y = batch
        preds = model(x.to(device))
        loss = criterion(preds, y.to(device))
        loss.backward()
        optimizer.step()
        loss_tracker.append(loss.item())
    #    if batch % args.log_interval == 0:
        wandb.log({"training loss": loss})
                
    print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")        
torch.save(model.state_dict(), 'trained_model.pt')
            
plt.plot(loss_tracker, '-')
plt.xlabel('Training step')
plt.ylabel('Training loss')
plt.savefig("training_curve.png")

            

print("Evaluating until hitting the ceiling")
# add any additional argument that you want
args = parser.parse_args(sys.argv[2:])
print(args)
        
# TODO: Implement evaluation logic here


test_set = CorruptMnist(train=False)
dataloader = torch.utils.data.DataLoader(test_set, batch_size=128)
        
correct, total = 0, 0
for batch in dataloader:
    x, y = batch
            
    preds = model(x.to(device))
    preds = preds.argmax(dim=-1)

    # Assume that `x` is a batch of images, `y` is the ground truth labels, and `preds` are the model's predictions
    wandb.log({"examples": [wandb.Image(x[i], caption=f'GT: {y[i]}  Pred: {preds[i]}') for i in range(len(x))]})

            
    correct += (preds == y.to(device)).sum().item()
    total += y.numel()
            
print(f"Test set accuracy {correct/total}")
wandb.log({"Test set accuracy ": correct/total})
