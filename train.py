import einops
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import torch
from tqdm import tqdm
import cv2
from config import PPGConfig, Mode
from data_loader import get_data_loaders
from metrics import Metrics
from model import PPGformer


class PPGClassifierTrainer:

    def __init__(self, config: PPGConfig) -> None:
        self.model = PPGformer(
            embed_size=config.model.embed_size,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            num_classes=config.model.num_classes,
            signal_length=config.model.signal_length,
            expansion=config.model.expansion,
            input_channels=config.model.input_channels
        ).to(config.device)
        self.config = config
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=1e-4)
        self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.1, 0.1]).to(self.config.device))
        self.data_loader = get_data_loaders(self.config.dataset)
        self.metrics = {
            Mode.train: Metrics(),
            Mode.eval: Metrics()
        }
        self.train_accuracy_history = []
        self.train_loss_history = []
        self.eval_accuracy_history = []
        self.eval_loss_history = []

    def train(self):
        confusion_matrices_image_train, confusion_matrices_image_eval = [], []
        for epoch in range(self.config.num_epochs):
            confusion_matrices_image_train.append(self.train_epoch(epoch))

            if epoch % self.config.validation_frequency == 0:
                confusion_matrices_image_eval.append(self.validate_epoch(epoch))
        return confusion_matrices_image_train, confusion_matrices_image_eval

    def train_epoch(self, epoch):
        self.model.train()
        loader = tqdm(self.data_loader[Mode.train])
        accuracy = 0
        total_loss = 0
        self.metrics[Mode.train].reset()
        for index, data in enumerate(loader):
            self.optimizer.zero_grad()
            signal, label = [d.to(self.config.device) for d in data]
            prediction = self.model(einops.rearrange(signal, "b c e -> b e c"))
            loss = self.loss(prediction, label)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            accuracy += torch.sum(prediction.argmax(1) == label).cpu().numpy()  # Transfert vers CPU
            self.metrics[Mode.train].update(prediction.argmax(1), label)
            loader.set_description(f"TRAINING: {epoch}, loss: {loss.item()}. Target: {label[:8].tolist()}, Prediction: {prediction.argmax(1)[:8].tolist()}")

        # Calcule de la précision et de la perte moyennes pour l'époch d'entraînement
        avg_loss = total_loss / len(loader)
        avg_accuracy = accuracy / len(loader) / self.config.dataset.batch_size
        print(f"TRAINING Loss: {avg_loss}, Accuracy: {avg_accuracy}")

        # Enregistrer l'historique de précision et de perte
        self.train_accuracy_history.append(avg_accuracy)
        self.train_loss_history.append(avg_loss)

        return self.metrics[Mode.train].confusion_matrix_image()

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()
        loader = tqdm(self.data_loader[Mode.eval])
        accuracy = 0
        total_loss = 0
        self.metrics[Mode.eval].reset()
        for index, data in enumerate(loader):
            signal, label = [d.to(self.config.device) for d in data]
            prediction = self.model(einops.rearrange(signal, "b c e -> b e c"))
            loss = self.loss(prediction, label)
            total_loss += loss.item()
            accuracy += torch.sum(prediction.argmax(1) == label).cpu().numpy()  # Transfert vers CPU
            self.metrics[Mode.eval].update(prediction.argmax(1), label)
            loader.set_description(f"VALIDATION: {epoch}, loss: {loss.item()}. Target: {label[:8].tolist()}, Prediction: {prediction.argmax(1)[:8].tolist()}")

        # Calcule de la précision et de la perte moyennes pour l'époch de validation
        avg_loss = total_loss / len(loader)
        avg_accuracy = accuracy / len(loader) / self.config.dataset.batch_size
        print(f"VALIDATION Loss: {avg_loss}, Accuracy: {avg_accuracy}")

        # Enregistrer l'historique de précision et de perte
        self.eval_accuracy_history.append(avg_accuracy)
        self.eval_loss_history.append(avg_loss)

        return self.metrics[Mode.eval].confusion_matrix_image()

    def plot_training_history(self):

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.config.num_epochs + 1), self.train_accuracy_history, label='Training Accuracy')
        plt.plot(range(1, self.config.num_epochs + 1, self.config.validation_frequency), self.eval_accuracy_history, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('accuracy_plot.png')  # Enregistrer la figure comme image

        plt.subplot(1, 2, 2)
        plt.plot(range(1, self.config.num_epochs + 1), self.train_loss_history, label='Training Loss')
        plt.plot(range(1, self.config.num_epochs + 1, self.config.validation_frequency), self.eval_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_plot.png')  # Enregistrer la figure comme image



if __name__ == "__main__":
    config = PPGConfig()
    trainer = PPGClassifierTrainer(config)

    train_confusion_matrix, eval_confusion_matrix = trainer.train()

    writer_train = cv2.VideoWriter("train.avi", cv2.VideoWriter_fourcc(*"XVID"), 1, (train_confusion_matrix[0].shape[1], train_confusion_matrix[0].shape[0]))
    writer_eval = cv2.VideoWriter("eval.avi", cv2.VideoWriter_fourcc(*"XVID"), 1, (eval_confusion_matrix[0].shape[1], eval_confusion_matrix[0].shape[0]))

    for cm in train_confusion_matrix:
        writer_train.write(cm)

    for cm in eval_confusion_matrix:
        writer_eval.write(cm)

    trainer.plot_training_history()