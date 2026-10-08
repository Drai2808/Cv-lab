import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
class Dashboard:
    def print_metrics(self, y_true, y_pred, output_img_path):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = rec
        print("=" * 40)
        print("CLASSIFICATION PERFORMANCE MEASURES")
        print("=" * 40)
        print(f"Accuracy:         {acc:.4f}")
        print(f"Precision:        {prec:.4f}")
        print(f"Sensitivity (SC): {sensitivity:.4f}")
        print(f"Specificity (SP): {specificity:.4f}")
        print(f"F1-Score:         {f1:.4f}")
        print("-" * 40)
        print("Confusion Matrix:")
        print(cm)
        print("-" * 40)
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal'], zero_division=0))
        print("=" * 40)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(['Normal', 'Abnormal'])
        axes[0].set_yticks([0, 1])
        axes[0].set_yticklabels(['Normal', 'Abnormal'])
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[0].text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black")
        metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score']
        values = [acc, prec, sensitivity, specificity, f1]
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
        bars = axes[1].bar(metrics, values, color=colors)
        axes[1].set_title('Performance Metrics')
        axes[1].set_ylim(0, 1.1)
        for bar in bars:
            yval = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(output_img_path)
        plt.close()
        print(f"\nResults image saved to: {output_img_path}")