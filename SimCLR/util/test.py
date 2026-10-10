import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os

class ModelEvaluator:
    def __init__(self, model, test_loader, device, log_parent_dir, log_dir):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.log_parent_dir = log_parent_dir
        self.log_dir = log_dir

    def evaluate_model(self):
        self.model.eval()
        correct = 0
        total = 0
        all_predicted = []
        all_labels = []

        progress_bar = tqdm(self.test_loader, desc='Testing', unit='batch')

        with torch.no_grad():
            for images, labels in progress_bar:

                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                progress_bar.set_postfix(accuracy=100 * correct / total)

        accuracy = correct / total
        print(f'Accuracy on test set: {100 * accuracy:.2f}%')

        conf_matrix = confusion_matrix(all_labels, all_predicted)
        class_report = classification_report(all_labels, all_predicted, output_dict=True, zero_division=1)


        report_df = pd.DataFrame(class_report).transpose()


        mean_precision = report_df.loc['macro avg', 'precision']
        mean_recall = report_df.loc['macro avg', 'recall']
        mean_f1 = report_df.loc['macro avg', 'f1-score']


        results = pd.DataFrame({
            'Accuracy': [accuracy * 100],
            'Mean Precision': [mean_precision * 100],
            'Mean Recall': [mean_recall * 100],
            'Mean F1 Score': [mean_f1 * 100]
        })


        results.to_csv(os.path.join(self.log_parent_dir, self.log_dir, 'results.csv'), index=False)

        conf_matrix_df = pd.DataFrame(conf_matrix)
        conf_matrix_df.to_csv(os.path.join(self.log_parent_dir, self.log_dir, 'confusion_matrix.csv'), index=False)
