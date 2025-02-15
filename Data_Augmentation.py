# importing all the required libraries
import warnings
warnings.filterwarnings('ignore')
import torch
import random
import numpy as np
from torchsummary import summary  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import plotext  # type: ignore
import torchvision.transforms as transforms

def data_augmentation(data_train_X, data_train_Y):
    # Setting the seed
    np.random.seed(69)

    # Creating and storing all the transformations
    transform = transforms.Compose([
       transforms.ColorJitter(brightness=(0.5, 1.5)),
       transforms.RandomAffine(degrees=0, translate=(0.2, 0), fill=0),
       transforms.RandomAdjustSharpness(sharpness_factor=random.sample(range(0, 3), 1)[0], p=0.3),
       transforms.RandomRotation(degrees=(0, 10))
    ])

    # Sampling a predefined number of images from the training dataset using reservoir sampling

    # Create a list of indices corresponding to the images in the original dataset
    indices = list(range(len(data_train_X)))

    # Setting the sample size
    k = 1000

    # Use the reservoir sampling algorithm to sample k images from the dataset
    reservoir = data_train_X[:k].copy()
    sampled_indices = indices[:k].copy()
    for i in range(k, len(data_train_X)):
       j = random.randint(0, i)
       if j < k:
           reservoir[j] = data_train_X[i].copy()
           sampled_indices[j] = indices[i]

    # Storing the sampled X and the sampled Y
    sampled_X = reservoir
    sampled_Y = data_train_Y[sampled_indices]

    # Create an empty numpy array to store augmented images
    augmented_images = np.empty_like(sampled_X)

    # Iterate over the sampled images, apply the transformation pipeline and store the augmented images
    for idx, image in enumerate(sampled_X):
       augmented_image = transform(torch.tensor(image)).numpy()
       augmented_images[idx] = augmented_image

    # Concatenate the original and augmented images along the first axis for X dataset
    all_images = np.concatenate((data_train_X, augmented_images), axis=0)

    # Concatenate the original Y dataset with the class labels for the augmented data
    all_Y = np.concatenate((data_train_Y, sampled_Y), axis=0)

    # Randomising the order of the indices so the augmented data is not all at the end.
    # original lists
    list1 = all_images
    list2 = all_Y

    # combine the two lists using zip
    combined = list(zip(list1, list2))
    # shuffle the combined list
    random.shuffle(combined)

    # separate the shuffled items back into their original lists
    shuffled_list1, shuffled_list2 = zip(*combined)
    final_X = np.array(shuffled_list1)
    final_Y = np.array(shuffled_list2)

    return final_X, final_Y

