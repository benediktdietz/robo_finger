# This File is being used for Post-Processing Raw Input and Output Data
# for generating the final version(s) of the Level 5 Dataset which will
# be used for training a regressor (i.e., CNN)

import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import datetime
import time

master_seed = 1244

########################################################################################
### LOAD THE RAW DATA
########################################################################################

# A path to all the .pkl files which contain the:
# Low Resolution Square Images and
# corresponding State Descriptions/Vectors
dir_raw_data = "/home/mzecevic/Desktop/level5_dataset_low_resolution_square"

list_path_data = sorted(glob.glob(os.path.join(dir_raw_data, "*.pkl")))
print("Found " + str(len(list_path_data)) + " (.pkl) Raw Data Files.")

# for development purposes take a smaller subset
developmental = False #True
if developmental:
    list_path_data = list_path_data[:50]
    print("***** DEVELOPMENTAL: USING ONLY A SUBSET, NOT WHOLE DATASET!!")

list_data = []
for ind, path_data in enumerate(list_path_data):
    if (ind+1) % 100 == 0:
        print("(Loaded " + str(ind+1) + "/" + str(len(list_path_data)) + " successfully.)")
    with open(path_data, "rb") as f:
        data = pickle.load(f)
        list_data.append(data)
assert(len(list_data) == len(list_path_data))
print("Loaded all data files successfully.")

num_datapoints = sum([len(d["input"]) for d in list_data])
print("Data Pairs available: " + str(num_datapoints))

#import pdb; pdb.set_trace()
########################################################################################
### CREATE THE DATASET(S)
########################################################################################

# Helper Function: saving e.g. a Dataset List using Pickle
def save_data_with_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    file_size_in_megabytes = os.path.getsize(path) >> 20
    print("Saved Dataset of " + str(len(data)) +
          " Datapoints (File Size in MB " + str(file_size_in_megabytes) +
          " to " + str(path) + ").")
# Helper Function: load e.g. a Pickle dumped Dataset
def load_data_with_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    file_size_in_megabytes = os.path.getsize(path) >> 20
    print("Loaded Dataset of " + str(len(data)) +
          " Datapoints (File Size in MB " + str(file_size_in_megabytes) + ").")
    return data
# Helper Function: displays an image
def show_image(img):
    plt.imshow(img)
    plt.show()
    plt.clf()
# Helper Function: Permutates the Data randomly
def randomly_shuffe_dataset(seed, dataset):
    # randomly permutate/shuffle the dataset
    indices_random_permutation_dataset = np.arange(0, len(dataset))
    np.random.seed(seed)
    np.random.shuffle(indices_random_permutation_dataset)
    dataset = [dataset[index] for index in indices_random_permutation_dataset] # dataset[indices_random_permutation_dataset.tolist()]
    print("Randomly Shuffled the Datapoints within the Dataset.")
    return dataset
# Helper Function: applies the transformations to the image
def dataset_images_transformation(x):
    torch_transformation_scheme = transforms.Compose([
        # not necessary given that the raw data is already squared low resolution # transforms.Resize((100,100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    if isinstance(x, np.ndarray):
        x = Image.fromarray(x)
    return torch_transformation_scheme(x)
    # This function transforms the dataset to two image inputs e.g. (6,100,100) and next step (i.e., second image data) targets
    # so that it can be used by the CNN to predict angle velocity as well
def transform_to_sequence_dataset(dataset):
    #assert() # make sure the number of data points is even for remainder less composition
    if len(dataset) % 2 != 0:
        dataset = dataset[:-1]
        print("Removed last element of Dataset to create a remainder-less Sequence Dataset.")
    assert(len(dataset) % 2 == 0)
    new_dataset = []
    for ind, data_tuple in enumerate(dataset):
        if ind % 2 == 1: # always two are being grouped together without overlap!
            continue
        if ind == (len(dataset)-1): # to handle the last index
            break
        new_input = torch.cat((data_tuple[0], dataset[ind + 1][0])) # passing two images, current and next
        new_input = torch.cat((new_input[:3,:], new_input[3:,:]), dim=2)
        new_target = dataset[ind + 1][1] # but predicting the next frame
        new_datapoint = (new_input, new_target)
        new_dataset.append(new_datapoint)
    print("Created the Sequence-Input-CNN Dataset with " + str(len(new_dataset)) + " Datapoints.")
    print("Used the concatenated as single Image opposed to overlapping channels representation.")
    return new_dataset

#------------------------------------------------------------------------------------------------------------

# create_sequence_dataset = True # If True, then pairs of images are going to be a single input (for Velocity inference)

reduced_state_space_dataset = True # If True, then only use following in target: angles, obj pos, tip pos, target pos

for bool_sequence in [False]: # running both in succession will probably be too RAM greedy

    if bool_sequence:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SEQUENCE (CONCATENATED DOUBLE INPUT) DATASET")
    else:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SIMPLE (SINGLE INPUT) DATASET")

    inputs_raw = np.array([item for sublist in [d["input"] for d in list_data] for item in sublist])
    targets_raw = np.array([item for sublist in [d["target"] for d in list_data] for item in sublist])
    assert(inputs_raw.shape[0] == targets_raw.shape[0])

    if reduced_state_space_dataset:

        # for reference please consider the definition of the full-state-space in finger_env.py
        # the original state is 22-dimensional, and with following procedure it is reduced to 12-dimensions
        # the remaining dimensions are: angles, obj pos, tip pos, goal pos
        targets_raw = [np.concatenate((t[3:6], t[9:12], t[16:])) for t in targets_raw]

    dataset = [(dataset_images_transformation(d), torch.Tensor(targets_raw[ind])) for ind, d in enumerate(inputs_raw)]
    print("Generated a Torch-Ready Dataset.")

    # if True, create the Sequence-Input-Dataset Variant
    if bool_sequence: # create_sequence_dataset:

        dataset = transform_to_sequence_dataset(dataset)

    # randomly shuffle the raw data
    dataset = randomly_shuffe_dataset(master_seed, dataset)
    #import pdb; pdb.set_trace()

    # create a train and test split
    percentage_of_test_from_whole_dataset = 0.1
    number_of_test_datapoints = int(len(dataset) * percentage_of_test_from_whole_dataset)
    dataset_train = [dataset[index] for index in range(0, len(dataset))[:-number_of_test_datapoints]] # dataset[:-number_of_test_datapoints, :]
    dataset_test = [dataset[index] for index in range(0, len(dataset))[-number_of_test_datapoints:]] # dataset[-number_of_test_datapoints:, :]
    print("Created a Train-Test Split with Ratio for Test being " + str(percentage_of_test_from_whole_dataset))
    print("Number of Training Points: " + str(len(dataset_train)))
    print("Number of Testing Points: " + str(len(dataset_test)))
    # clean up RAM
    del dataset
    del inputs_raw
    del targets_raw

    # save the dataset
    print("Saving Datasets Now...")
    start = time.time()
    file_name = "level5_dataset_"
    if bool_sequence: # create_sequence_dataset:
        file_name = file_name + "sequence_"
    if reduced_state_space_dataset:
        file_name = file_name + "reduced_"
    if not bool_sequence and not reduced_state_space_dataset:
        file_name = file_name + "simple_"
    if developmental:
        file_name = file_name + "subset_"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M_")
    save_data_with_pickle(timestamp + file_name + "train.pkl", dataset_train)
    saving_time = (time.time() - start) / 60
    print("Saving took " + str(int(saving_time)) + " minutes.")
    save_data_with_pickle(timestamp + file_name + "test.pkl", dataset_test)
    saving_time = (time.time() - start) / 60
    print("Saving took " + str(int(saving_time)) + " minutes.")
