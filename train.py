import torch 
import torch.nn as nn
from model import Model
from MNIST_data_loader import load_MNIST

model = Model()
train_dataLoader = load_MNIST()[0]

optimizer1 = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_func = nn.CrossEntropyLoss()


def train_model(model_train, train_loader, optimizer, loss_function, epochs):
    model_train.train()
    for i in range(epochs):
        total_loss=0
        for images, labels in train_loader:
            model_train.zero_grad()
            output = model_train(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss

        print(f"Epoch {i}: {total_loss}")



train_model(model, train_dataLoader, optimizer1, loss_func, 10)
torch.save(model.state_dict(), "saved_models/mnist_model.pth")
        

