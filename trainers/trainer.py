import torch
from tqdm import tqdm
from losses.l1 import l1_loss
from losses.silog import silog_loss

class Trainer:
    def __init__(self, model, loader, optimizer, logger, device):
        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.logger = logger
        self.device = device

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for img, depth in tqdm(self.loader):
                img, depth = img.to(self.device), depth.to(self.device)
                pred = self.model(img)

                loss = l1_loss(pred, depth) + 0.5 * silog_loss(pred, depth)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            self.logger.log({
                "epoch": epoch,
                "loss": total_loss / len(self.loader)
            })