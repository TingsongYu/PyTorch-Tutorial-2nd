import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
plt.switch_backend('agg')

def plot_confusion_matrix(y_true, y_pred, classes,
                          save_path,normalize=False,title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # --- plot--- #
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.figsize'] = [20, 20]  # plot
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # --- bar --- #
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    # --- bar --- #
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_path)

if __name__ == "__main__":
    y_true = ['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER', 'O']
    y_pred = ['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O','B-PER', 'I-PER', 'O']
    classes = ['O','B-MISC', 'I-MISC','B-PER', 'I-PER']
    save_path = './ner_confusion_matrix.png'
    plot_confusion_matrix(y_true,y_pred,classes,save_path)