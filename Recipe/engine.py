"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm


def train_step(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        train_dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """
    # setting the model to train mode
    total_loss, total_acc = 0, 0
    for i, (X, y) in enumerate(train_dataloader):
        # to device
        X, y = X.to(device), y.to(device)
        # forward pass
        y_logits = model(X)
        # loss calculate
        loss = loss_fn(y_logits, y)
        # zero the optimizer
        optimizer.zero_grad()
        # loss backward
        loss.backward()
        # optimizer step
        optimizer.step()
        total_acc += (torch.argmax(y_logits, dim=1) == y).sum() / len(y)
        total_loss += loss
    total_loss /= len(train_dataloader)
    total_acc = total_acc / len(train_dataloader)
    return total_loss, total_acc


def test_step(model: torch.nn.Module,
              test_dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        test_dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
    """
    # eval model
    with torch.inference_mode():
        test_loss, test_acc = 0, 0
        for i, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            test_acc += (torch.argmax(y_logits, dim=1) == y).sum() / len(y)
            test_loss += loss_fn(y_logits, y)
        test_loss /= len(test_dataloader)
        test_acc = test_acc / len(test_dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device):
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for
        each epoch.
        In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]}
        For example if training for epochs=2:
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]}
                  
    Remark : Can un-comment the below lines of code to also store the model weights 
    """
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
    #     from pathlib import Path
    #     save_path=Path("models")
    #     save_path.mkdir(parents=True,exist_ok=True)
    model.to(device)
    for epoch in tqdm(range(epochs)):
        model.train()
        loss_train, acc_train = train_step(model,
                                           train_dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        results["train_loss"].append(loss_train)
        results["train_acc"].append(acc_train)
        model.eval()
        loss_test, acc_test = test_step(model,
                                        test_dataloader,
                                        loss_fn,
                                        device)
        results["test_loss"].append(loss_test)
        results["test_acc"].append(acc_test)
        print(
            f"epoch : {epoch + 1} | train_loss : {loss_train} | train_acc : {acc_train} | test_loss : {loss_test} | test_acc : {acc_test}")
    #         model_path=save_path/f"model_efficient_b2_weights_{epoch}.pth"
    #         torch.save(obj=model.state_dict(),f=model_path)
    return results
