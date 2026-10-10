import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.optim import lr_scheduler
from thop import profile

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.scheduler = scheduler

    def train_model(self, num_epochs, log_parent_dir, log_dir):

        os.makedirs(os.path.join(log_parent_dir, log_dir), exist_ok=True)
        log_file_path = os.path.join(log_parent_dir, log_dir, 'training_logs.log')
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batches')

            for i, (images, labels) in progress_bar:

                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                running_loss += loss.item()

                progress_bar.set_postfix(loss=running_loss/(i+1), accuracy=100 * correct / total)

            self.train_losses.append(running_loss/len(self.train_loader))
            self.train_accuracies.append(100 * correct / total)

            if self.scheduler is not None:
                self.scheduler.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(self.train_loader)}, Accuracy: {100 * correct / total}%')

            self.model.eval()
            correct = 0
            total = 0

            progress_bar = tqdm(self.val_loader, desc='Validation', unit='batch')

            with torch.no_grad():
                for images, labels in progress_bar:

                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    progress_bar.set_postfix(accuracy=100 * correct / total)

            val_accuracy = correct / total
            self.val_accuracies.append(100 * val_accuracy)
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {100 * val_accuracy:.2f}%')

            model_path = os.path.join(log_parent_dir, log_dir, 'model.pth')
            torch.save(self.model.state_dict(), model_path)


            fig, ax1 = plt.subplots(figsize=(12, 6))

            color = 'tab:red'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss', color=color)
            line1 = ax1.plot(self.train_losses, color=color, label='Training Loss')
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Accuracy (%)', color=color)
            line2 = ax2.plot(self.train_accuracies, color=color, label='Training Accuracy')
            line3 = ax2.plot(self.val_accuracies, color='tab:green', label='Validation Accuracy')
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title('Training Progress')
            fig.tight_layout()
            lines = line1 + line2 + line3
            labels = [line.get_label() for line in lines]
            ax2.legend(lines, labels, loc='upper left')

            plt.savefig(os.path.join(log_parent_dir, log_dir, 'training_progress.png'))
            plt.close(fig)


            with open(log_file_path, 'a') as f:
                f.write(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {self.train_losses[epoch]:.4f}, '
                    f'Training Accuracy: {self.train_accuracies[epoch]:.2f}%, '
                    f'Validation Accuracy: {100 * val_accuracy:.2f}%\n')
