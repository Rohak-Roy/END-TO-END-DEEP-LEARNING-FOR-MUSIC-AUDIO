# from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from dataset import MagnaTagATune
from torch.utils.data import DataLoader
from evaluation import evaluate

# from base_model import Model
from improved_model import Model

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

LENGTH = 256
STRIDE = 256
TRAIN_DATA_PATH = 'MagnaTagATune/annotations/train_labels.pkl'
VAL_DATASET_PATH = 'MagnaTagATune/annotations/val_labels.pkl'
SAMPLES_PATH = 'MagnaTagATune/samples_norm'

training_data = MagnaTagATune(TRAIN_DATA_PATH, SAMPLES_PATH)
train_dataloader = DataLoader(training_data, batch_size=10, shuffle=True, num_workers=10)

val_data = MagnaTagATune(VAL_DATASET_PATH, SAMPLES_PATH)
val_dataloader = DataLoader(val_data, num_workers=10)

model = Model(LENGTH, STRIDE)
model.to(device)

# HYPERPARAMETERS ------------------------------------------------------------------------
EPOCHS = 20                                                        
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)                  
criterion = nn.BCELoss() 
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.2)                                                              
# ----------------------------------------------------------------------------------------

AUC_log = []
train_loss_log = []
epoch_log = []
best_AUC_score = float('-inf')
minibatch_log = []
minibatch_loss_log = []
minibatch_number = 0
# writer = SummaryWriter()

for epoch in range(EPOCHS):
  print(f'Starting Epoch: {epoch + 1}')

  running_loss = 0

  for batch_idx, data in enumerate(train_dataloader):
    _, batch, labels = data
    batch = batch.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = model(batch)

    loss = criterion(outputs, labels) 

    loss.backward()

    optimizer.step()

    running_loss += loss.item()

    minibatch_number += 1
    if minibatch_number % 300 == 299:
      minibatch_log.append(minibatch_number)
      minibatch_loss_log.append(round(loss.item(), 4))
    #   writer.add_scalar('Mini-batch Loss', round(loss.item(), 4), minibatch_number)

    if batch_idx % 10 == 9:
        epoch_num = epoch + 1
        actual_loss  = running_loss / 10
        running_loss = 0

  model.eval()
  with torch.no_grad():
    all_preds = []

    for idx, data in enumerate(val_dataloader):
      _, val_data, labels = data
      val_data = val_data.to(device)

      preds = model(val_data)
      preds = torch.flatten(preds, start_dim=0)     # [1, 50] --> [50]
      all_preds.append(preds)

    AUC_score = round(evaluate(all_preds, VAL_DATASET_PATH), 4)

  model.train()

  lr_scheduler.step(AUC_score)

  if(AUC_score > best_AUC_score):
    best_AUC_score = AUC_score
    torch.save(model.state_dict(), 'best_model.pth.tar')

  print(f'At end of epoch number {epoch + 1}: TRAINING LOSS = {actual_loss}, AUC SCORE = {AUC_score}', flush=True)  
  train_loss_log.append(actual_loss)
  AUC_log.append(AUC_score)
  epoch_log.append(epoch_num)
#   writer.add_scalar('Train Loss', round(actual_loss, 4), epoch)
#   writer.add_scalar('AUC score', AUC_score, epoch)

# writer.close()