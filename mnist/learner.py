import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader


class BasicOptim:
    def __init__(self, params, lr):
        self.params, self.lr = list(params), lr

    def set_lr(self, lr):
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= p.grad.data * self.lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class Learner:
    def __init__(
        self, dl: DataLoader, model, opt_func, loss_func, label_map, lr=0.01, log=False
    ):
        self.dl = dl
        self.model = model
        self.loss_func = loss_func
        self.lr = lr
        self.opt = opt_func(self.model.parameters(), self.lr)
        self.log = log
        self.label_map = label_map

    def get_loss(self, preds, yb):
        return self.loss_func(preds, yb)

    def calc_grad(self, xb, yb):
        preds = self.model(xb)
        loss = self.loss_func(preds, yb, self.label_map)
        loss.backward()
        return loss

    def calc_accuracy(self, preds, targets):
        """Calculate accuracy for a batch"""
        # Get predicted class (argmax of logits)
        predicted_classes = torch.argmax(preds, dim=1)
        # Compare with true targets
        correct = (predicted_classes == targets).float()
        return correct.mean().item()

    def train_epoch(self, epoch_num=None, should_print=False):
        self.model.train()
        accumulated_loss = 0
        accumulated_acc = 0
        num_batches = 0

        for xb, yb in self.dl:
            loss = self.calc_grad(xb, yb)
            self.opt.step()
            self.opt.zero_grad()

            if should_print:
                with torch.no_grad():
                    preds = self.model(xb)
                accumulated_loss += loss.item()
                accumulated_acc += self.calc_accuracy(preds, yb)
                num_batches += 1

        if should_print:
            avg_loss = accumulated_loss / num_batches
            avg_acc = accumulated_acc / num_batches
            print(f"Epoch {epoch_num}: L {avg_loss:.4f} | Acc {avg_acc:.4f}")

    def train_model(self, epochs, lr=None, print_times=5):
        if lr is not None:
            self.opt.set_lr(lr)

        # Calculate which epochs to print (evenly spaced, always print last epoch)
        print_epochs = set(
            [
                int(round(i * (epochs - 1) / (print_times - 1)))
                for i in range(print_times)
            ]
        )

        for epoch in range(epochs):
            epoch_num = epoch + 1
            should_print = epoch in print_epochs
            self.train_epoch(epoch_num, should_print)

    def predict(self, x: torch.Tensor, batch_size=256):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                xb = x[i : i + batch_size]
                out = self.model(xb)
                pred_labels = torch.argmax(out, dim=1)
                preds.append(pred_labels.cpu())
        return torch.cat(preds)


def cross_entropy_loss(
    predictions: Tensor, targets: Tensor, label_map, reduction="mean"
):
    # Apply softmax to convert logits to probabilities
    probs = torch.softmax(predictions, dim=1)

    mapped_targets = label_map[targets]

    # Convert targets to one-hot encoding
    batch_size = targets.size(0)
    num_classes = predictions.size(1)
    targets_one_hot = torch.zeros(batch_size, num_classes, dtype=targets.dtype)
    targets_one_hot.scatter_(1, mapped_targets.unsqueeze(1), 1)

    # Calculate cross-entropy loss: -sum(target * log(prob))
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    loss = -torch.sum(targets_one_hot * torch.log(probs + epsilon), dim=1)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
