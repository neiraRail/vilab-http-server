import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class History:
    def __init__(self):
        self.train = {"loss": []}
        self.test = {"loss": []}

    def saveTrainLoss(self, loss):
        self.train["loss"].append(loss)

    def saveTestLoss(self, loss):
        self.test["loss"].append(loss)

    def saveTrainTestLoss(self, train, test):
        self.saveTrainLoss(train)
        self.saveTestLoss(test)

    def saveTrainMetric(self, metric, value):
        if not metric in self.train:
            self.train[metric] = []
        self.train[metric].append(value)

    def saveTestMetric(self, metric, value):
        if not metric in self.test:
            self.test[metric] = []
        self.test[metric].append(value)

    def saveTrainTestMetric(self, metric, train, test):
        self.saveTrainMetric(metric, train)
        self.saveTestMetric(metric, test)

    def printLast(self):
        print(f"Training. Loss: {self.train['loss'][-1]:>6f}, ", end="")
        if "accuracy" in self.train:
            print(f"Accuracy: {self.train['accuracy'][-1]:>6f} || ", end="")
        else:
            print("|| ", end="")
        print(f"Testing. Loss: {self.test['loss'][-1]:>6f}, ", end="")
        if "accuracy" in self.test:
            print(f"Accuracy: {self.test['accuracy'][-1]:>6f}",)
        else:
            print("")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

# Function to save model checkpoint
def save_checkpoint(model, optimizer, epoch, loss, file_path="best_model.pth"):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(state, file_path)


# Updated train loop with early stopping and checkpointing
def train_classifier(model, dataloader, loss_fn, optimizer, device, early_stopping=None, save_best=False):
    n_batches = len(dataloader)
    size = len(dataloader.dataset)

    model.train()
    correct = 0
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # Send X to device (GPU or CPU)
        X = X.to(device)
        y = y.to(device)

        # Prediction
        pred = model(X)

        # Loss function
        loss = loss_fn(pred, y.type(torch.int64))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Calculate average loss for the epoch
    train_loss /= n_batches
    accuracy = correct / size

    if early_stopping:
        if early_stopping(train_loss):
            print("Early stopping triggered.")
            return train_loss, True

    return (train_loss, accuracy), False


def test_classifier(dataloader, model, loss_fn, device):
  n_batches = len(dataloader)
  size = len(dataloader.dataset)

  model.eval()
  total_loss = []
  test_loss, correct = 0, 0
  with torch.no_grad():
    for batch, (X,y) in enumerate(dataloader):
      X = X.to(device)
      y = y.to(device)
      pred = model(X)

      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
      test_loss += loss_fn(pred, y.type(torch.int64)).item()

    correct /= size
    test_loss /= n_batches


  return test_loss, correct


# Updated train loop with early stopping and checkpointing
def train_ae(model, dataloader, loss_fn, optimizer, early_stopping=None, save_best=False, device="cuda"):
    n_batches = len(dataloader)
    size = len(dataloader.dataset)

    model.train()
    correct = 0
    train_loss = 0
    for batch, X in enumerate(dataloader):
        # Send X to device (GPU or CPU)
        X = X.to(device)

        # Prediction
        pred = model(X)

        # Loss function
        loss = loss_fn(pred, X)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    # Calculate average loss for the epoch
    train_loss /= n_batches

    if early_stopping:
        if early_stopping(train_loss):
            print("Early stopping triggered.")
            return train_loss, True

    return train_loss, False


def train_predictor(model, dataloader, loss_fn, optimizer, early_stopping=None, save_best=False):
    n_batches = len(dataloader)
    size = len(dataloader.dataset)

    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # Send X to device (GPU or CPU)
        X = X.to(device)
        y = y.to(device)

        # Prediction
        pred = model(X)

        # Loss function
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    # Calculate average loss for the epoch
    train_loss /= n_batches

    if early_stopping:
        if early_stopping(train_loss):
            print("Early stopping triggered.")
            return train_loss, True

    return train_loss, False



## Example usage of everitything above with an example model
# seq_len = 100
# n_features = 6
# learning_rate = 0.001
# embedding_dim = 128
# batch_size = 16

# model = VAECNNAutoencoder(4)
# model = model.to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# loss_fn = nn.MSELoss().to(device)

# # Training loop with history, checkpointing, and early stopping
# epochs = 5
# history = History()
# early_stopping = EarlyStopping(patience=20, min_delta=0.0001)
# actual_datetime = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")
# model_path=f"/content/drive/MyDrive/AAFE/autoencoder_{actual_datetime}_best.pth"
# best_loss = float('inf')

# for epoch in range(epochs):
#     print(f"Epoch {epoch+1}\n-------------------------------")

#     # Train the model
#     train_loss, stop = train_ae(model, train_dataloader, loss_fn, optimizer, early_stopping=early_stopping)

#     # test AE
#     test_loss = 0
#     model.eval()
#     with torch.no_grad():
#         for batch in test_dataloader:
#             x = batch
#             x = x.to(device)
#             ypred = model(x)
#             loss = loss_fn(ypred, x)
#             test_loss += loss

#     test_loss /= len(test_dataloader)

#     # Save train loss history
#     history.saveTrainTestLoss(train_loss, test_loss)  # Assuming no validation loss here
#     history.printLast()

