import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import TrainDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from vdsr import Net

# Parameters
train_epoch = 50
step = 10  # Sets the learning rate to the initial LR decayed by momentum every 10 epochs
batchSize = 64
learning_rate = 0.1
momentum = 0.9
weight_decay = 1e-4
clip = 0.4
# Constants
TRAIN_PATH = 'training_hr_images'


# Funtions for training
def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = learning_rate * (0.1 ** (epoch // step))
    return lr


def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()

    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader):
        # Put data to gpu if possible
        inputs, targets = Variable(lr_tensor), Variable(
            hr_tensor, requires_grad=False)
        inputs = inputs.cuda().float()
        targets = targets.cuda().float()

        # Forward
        loss = criterion(model(inputs), targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

    print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch,
                                                        iteration, len(training_data_loader), loss.data))


print("===> Loading datasets")
train_set = TrainDataset(TRAIN_PATH)
train_loader = DataLoader(
    dataset=train_set, num_workers=0, batch_size=batchSize, shuffle=True)

print("===> Building model")
model = Net()
criterion = nn.MSELoss(size_average=False)
model = model.cuda()
criterion = criterion.cuda()
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum, weight_decay=weight_decay)

print("===> Training")
for epoch in range(train_epoch):
    train(train_loader, optimizer, model, criterion, epoch)
# Save trained model
if not os.path.exists('trained_model/'):
    os.makedirs('trained_model')
torch.save(model, 'trained_model/'+'model_epoch_{}.pth'.format(train_epoch))
