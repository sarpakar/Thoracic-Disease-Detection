# Custom imports
from sklearn.model_selection import train_test_split

from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net
from dc1.train_test import train_model, test_model
from Data_Augmentation import data_augmentation
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from dc1.Evaluation_matrix import evaluate
# Torch imports
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import os
import argparse
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List


def main(args: argparse.Namespace, activeloop: bool = True) -> None:

    # Loading the datasets
    X_train = np.load("data/X_train.npy")
    Y_train = np.load("data/Y_train.npy")

    # Augment the training datasets
    print("Augmenting the training data...")
    X_train, Y_train = data_augmentation(X_train, Y_train)

    # Split data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.5, random_state=42)

    # Stratify data and flatten it to 2D arrays
    rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)
    X_rus_train, Y_rus_train = rus.fit_resample(X_train.reshape(X_train.shape[0], -1), Y_train)
    X_rus_val, Y_rus_val = rus.fit_resample(X_val.reshape(X_val.shape[0], -1), Y_val)

    # Reshape the arrays into original shape
    X_rus_train = X_rus_train.reshape(-1, 1, 128, 128)
    X_rus_val = X_rus_val.reshape(-1, 1, 128, 128)
    Y_rus_train = Y_rus_train.flatten()
    Y_rus_val = Y_rus_val.flatten()

    # Save split data to disk
    np.save("data/X_train_split.npy", X_rus_train)
    np.save("data/Y_train_split.npy", Y_rus_train)
    np.save("data/X_val.npy", X_rus_val)
    np.save("data/Y_val.npy", Y_rus_val)

    # Create ImageDataset objects from the split data
    train_dataset = ImageDataset(Path("data/X_train_split.npy"), Path("data/Y_train_split.npy"))
    val_dataset = ImageDataset(Path("data/X_val.npy"), Path("data/Y_val.npy"))

    # Loading test data
    test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))

    # Load the Neural Net. NOTE: set number of distinct labels here
    model = Net(n_classes=6)

    # Initialize optimizer(s) and loss function(s)
    optimizer_name = 'SGD' #state the optimizer name
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.98))
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1) #run the default optimizer if not stated anything else
    loss_function = nn.CrossEntropyLoss()

    # fetch epoch and batch count from arguments
    n_epochs = args.nb_epochs
    batch_size = args.batch_size

    # IMPORTANT! Set this to True to see actual errors regarding
    # the structure of your model (GPU acceleration hides them)!
    # Also make sure you set this to False again for actual model training
    # as training your model with GPU-acceleration (CUDA/MPS) is much faster.
    DEBUG = False

    # Moving our model to the right device (CUDA will speed training up significantly!)
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)
    elif (
        torch.backends.mps.is_available() and not DEBUG
    ):  # PyTorch supports Apple Silicon GPU's from version 1.12
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
        model.to(device)
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)

    # Let's now train, validate and test our model for multiple epochs:
    train_sampler = BatchSampler(
        batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches
    )
    test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=args.balanced_batches
    )
    val_sampler = BatchSampler(
        batch_size=100, dataset=val_dataset, balanced=args.balanced_batches
    )

    best_val_loss = float("inf")   # smallest valuation loss over epochs
    patience = 5  # number of epochs to wait for improvement
    wait = 0  # number of epochs without improvement

    mean_losses_train: List[torch.Tensor] = []
    mean_losses_test: List[torch.Tensor] = []
    mean_losses_val: List[torch.Tensor] = []

    conf_matrix = torch.zeros((6, 6), dtype=torch.int64)

    for e in range(n_epochs):
        if activeloop:
            # Training:
            losses = train_model(model, train_sampler, optimizer, loss_function, device)

            # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss)
            print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}/n")

            # Validation
            losses = test_model(model, val_sampler, loss_function, device)

            # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_val.append(mean_loss)
            print(f"\nEpoch {e + 1} validation done, loss on validation set: {mean_loss}\n")

            ### Check for early stopping
            if mean_loss < best_val_loss:
                best_val_loss = mean_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"No improvement for {patience} epochs. Stopping early.")
                    break

            # Testing:
            losses = test_model(model, test_sampler, loss_function, device)

            label_names = ['Atelectasis', 'Effusion', 'Infiltration', 'No Finding', 'Nodule', 'Pneumothorax']

            y_true = []
            y_pred = []
            for batch in test_sampler:
                X, y = batch
                X = X.to(device)
                y = y.to(device)
                out = model(X)
                pred = out.argmax(dim=1)
                y_true.append(y.cpu())
                y_pred.append(pred.cpu())
                # Update confusion matrix
                conf_matrix += confusion_matrix(y.cpu(), pred.cpu(), labels=[0, 1, 2, 3, 4, 5])

            # # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_test.append(mean_loss)
            print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}\n")

            ### Plotting during training
            plotext.clf()
            plotext.scatter(mean_losses_train, label="train")
            plotext.scatter(mean_losses_test, label="test")
            plotext.scatter(mean_losses_val, label="val")
            plotext.title("Train, Validation and Test loss")

            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

            plotext.show()

    # Evaluation Matrix for each class
    eval_fig = evaluate(conf_matrix)

    # Plot the confusion matrix
    plt.figure(figsize=(13, 13))
    sns.set(font_scale=1.4)
    conf_fig = sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16}, cmap="Blues", fmt="d", xticklabels=label_names, yticklabels=label_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # retrieve current time to label artifacts
    now = datetime.now()

    # check if model_weights/ subdir exists
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))

    # Saving the model
    torch.save(model.state_dict(), f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt")

    # Create plot of losses
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

    ax1.plot(range(1, len(mean_losses_train)+1), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    ax2.plot(range(1, len(mean_losses_test)+1), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
    ax3.plot(range(1, len(mean_losses_val)+1), [x.detach().cpu() for x in mean_losses_val], label="Validation", color="green")
    fig.legend()

    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of losses
    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day: 02}_{now.hour}_{now.minute:02}.png")
    eval_fig.figure.savefig(Path("artifacts") / f"eval_{now.month:02}_{now.day: 02}_{now.hour}_{now.minute:02}.png")
    conf_fig.figure.savefig(Path("artifacts") / f"conf_{now.month:02}_{now.day: 02}_{now.hour}_{now.minute:02}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_epochs", help="number of training iterations", default=100, type=int
    )
    parser.add_argument("--batch_size", help="batch_size", default=65, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        default=True,
        type=bool,
    )
    args = parser.parse_args()

    main(args)
