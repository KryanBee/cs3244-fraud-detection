import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

def plot_roc(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")  # Random classifier baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(save_path)
    plt.close()

def plot_pr(y_true, y_prob, save_path):
    # Precision-recall curve (useful for imbalanced data)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(save_path)
    plt.close()