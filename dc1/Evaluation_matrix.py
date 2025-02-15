import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

class_dict = {0: 'Atelectasis',
1: 'Effusion',
2: 'Infiltration',
3: 'No Finding',
4: 'Nodule',
5: 'Pneumothorax'}

def evaluate(confusion_matrix):

    cm = confusion_matrix.numpy()
    # calculate precision, recall, and f1-score for each class
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # create evaluation matrix for each class
    eval_matrices = []
    for i in range(cm.shape[0]):
        eval_matrix = np.zeros((3, 1))
        eval_matrix[0] = precision[i]
        eval_matrix[1] = recall[i]
        eval_matrix[2] = f1_score[i]
        eval_matrices.append(eval_matrix)
    print(eval_matrices)

    fig1, axs = plt.subplots(nrows=1, ncols=cm.shape[0], figsize=(10, 5), sharey=True)
    fig1.suptitle('Evaluation Matrices for Each Class')

    for i in range(cm.shape[0]):
        ax = sns.heatmap(eval_matrices[i], annot=True, cmap='Blues', fmt='.2f',
                         xticklabels=[class_dict[i]], yticklabels=['precision', 'recall', 'f1-score'],
                         cbar=False, ax=axs[i])

    plt.show()
    return fig1