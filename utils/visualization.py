from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

def confusion_matrix_visualization(clf, X_test, y_test):
    plot = plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues, normalize=None)
    plot.ax_.set_title('confusion matrix')
    plt.show()