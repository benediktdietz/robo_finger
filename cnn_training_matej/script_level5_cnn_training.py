# This File is being used to train a Regressor i.e., CNN to predict
# the State Descriptions from Images to be able to handle Level 5 Tasks

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

number_of_training_iterations = 500

path_dataset_train = "20191017_level5_sequence_train_45k.pkl" #"20191009_1702_level5_dataset_sequence_train.pkl" # now directly on angelo # "20191009_1636_level5_dataset_sequence_subset_train.pkl"
path_dataset_test = "20191017_level5_sequence_test_5k.pkl"
path_dataset_valid = "20191017_level5_sequence_valid_5k.pkl"

path_checkpoint = None # now directly on angelo # "20191012_1651_level5_CNN_model/20191013_0112_checkpoint_epoch_200.th"
checkpoint_frequency = 10
print("Checkpoint Save Frequency set to: " + str(checkpoint_frequency))

validate_only = False # If True, then don't train just evaluate on Test set
validate_during_training = True # If True, then validation (best with a validation set) is also performed during training

dict_state_components = {
     "torques": (-22,-19),
     "angles": (-19,-16),
     "velocity": (-16,-13),
     "obj_pos": (-13,-10),
     "obj_rot": (-10,-6),
     "tip_pos": (-6,-3),
     "target_pos": (-3,None),
}

prediction_target = "velocity"

print(">>> PREDICTION TARGET is: " + str(prediction_target))

if prediction_target == "obj_rot":
    num_outputs = 4
else:
    num_outputs = 3

prediction_range_left,prediction_range_right = dict_state_components[prediction_target]

model_batch_size = 4

########################################################################################
### PREPARE THE TORCH DATASETS (TRAIN/TEST)
########################################################################################

# Helper Function: load e.g. a Pickle dumped Dataset
def load_data_with_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    file_size_in_megabytes = os.path.getsize(path) >> 20
    print("Loaded Dataset of " + str(len(data)) +
          " Datapoints (File Size in MB " + str(file_size_in_megabytes) + ").")
    return data
# Helper Function:
def get_dataset_loader(path_dataset_test,subset_cap=None):
    dataset_test = load_data_with_pickle(path_dataset_test)
    # adapt by only using the pushing target position for the Label/Target information
    dataset_test = [(d[0], d[1][prediction_range_left:prediction_range_right]) for ind, d in enumerate(dataset_test)]
    if subset_cap:
        dataset_test = dataset_test[:subset_cap]
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=model_batch_size, shuffle=True, num_workers=0)
    print("Loaded " + str(len(dataset_test)) + " Datapoints.")
    len_dataset = len(dataset_test)
    len_loader = len(loader_test)
    del dataset_test

    return loader_test, len_dataset, len_loader
# Function to validate a Model
def validate_model(model, path_dataset_test, display_test_results=True):

    loader_test, len_dataset, len_loader = get_dataset_loader(path_dataset_test)

    test_cost = 0
    n_examples = 5
    np.random.seed(master_seed)
    random_indices = np.random.randint(0, len_loader, n_examples) # show five random examples
    #import pdb; pdb.set_trace()
    print("Showing Random Examples Indices: " + str(sorted(list(random_indices))))
    for i, data in enumerate(loader_test, 0):
        test_inputs, test_labels = data
        predictions_test_batch = model(test_inputs)
        if display_test_results and i in random_indices:
            for ind, d in enumerate(predictions_test_batch):
                np.random.seed(master_seed)
                random_index_in_batch = np.random.randint(0, model_batch_size)
                if ind == random_index_in_batch:
                    print("-------")
                    print("Prediction " + str(ind) + ": " + str(d.tolist()) +
                      "\nGround Truth " + str(ind) + ": " + str(test_labels[ind].tolist()))
                    print("-------")
        test_cost += model_cost(predictions_test_batch, test_labels)
    print("********** Cost on Test Set " + str(test_cost.item()))
    print("********** Average Cost per Batch " + str(test_cost.item() / len_loader))
    del loader_test
    return test_cost


if not validate_only:

    subset_cap = None #30000
    loader_train, len_dataset_train, len_loader_train = get_dataset_loader(path_dataset_train, subset_cap)

    print("Using " + str(subset_cap) + " Datapoints for Memory Allocation Reasons.")

########################################################################################
### TRAIN THE NETWORK
########################################################################################

# Architecture of the CNN i.e., layers and activation functions
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # (3,128,128) -> (6,128,128)
        self.pool = nn.MaxPool2d(2, 2) # (6,128,128) -> (6,62,62)
        self.conv2 = nn.Conv2d(6, 16, 5) # (6,62,62) -> (16,29,29)
        self.fc1 = nn.Linear(16 * 29 * 61, 128) # reshape
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_outputs) # targets are 22-dimensional

    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = self.pool(F.relu(self.conv1(x)))
        #import pdb; pdb.set_trace()
        x = self.pool(F.relu(self.conv2(x)))
        #import pdb; pdb.set_trace()
        x = x.view(-1, 16 * 29 * 61)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = CNN()

model_cost = nn.MSELoss()

model_optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

continue_from_epoch = None
if path_checkpoint is not None:
   checkpoint = torch.load(path_checkpoint)
   model.load_state_dict(checkpoint['model_state_dict'])
   model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
   continue_from_epoch = checkpoint["epoch"]
   print("Continuing training of existing Model from Epoch " + str(continue_from_epoch) + " on.")

   if validate_only:
       print("Only validating model.")
       valid_cost = validate_model(model, path_dataset_valid, display_test_results=True)
       test_cost = validate_model(model, path_dataset_test, display_test_results=True)
       exit()

print("Starting training of the Model for " + str(number_of_training_iterations) + " Epochs.")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M_")

if path_checkpoint is not None:
   dir_experiment = os.path.dirname(path_checkpoint)
else:
   dir_experiment = timestamp + prediction_target + "_level5_CNN_model/"

name_performance_epoch = "performance_epoch_loss.log"
name_performance_validation = "performance_validation_loss.log"

time_start_training = time.time()
list_loss_training = []
last_epoch = 1
if continue_from_epoch is not None:
   epoch_start = continue_from_epoch
   last_epoch = continue_from_epoch
else:
   epoch_start = 1
for epoch in range(epoch_start,epoch_start + number_of_training_iterations + 1):  # loop over the dataset multiple times

    epoch_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(loader_train, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        model_optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        cost = model_cost(outputs, labels)
        cost.backward()
        model_optimizer.step()

        # print statistics
        running_loss += cost.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            if last_epoch != epoch:
                mode_flush = False
            else:
                mode_flush = True
            print("[Epoch %d, Batch %5d] Cost: %.6f" %
                  (epoch, i + 1, running_loss / 10), end="\r", flush=mode_flush)
            epoch_loss += running_loss
            running_loss = 0.0
            last_epoch = epoch
    print("[Epoch %d] Overall Loss: %.6f                     ---" % (epoch, epoch_loss))
    list_loss_training.append(epoch_loss)

    if not os.path.exists(dir_experiment):
        os.makedirs(dir_experiment)

    if epoch % checkpoint_frequency == 0:
        
        torch.save({
           'epoch': epoch,
           'model_state_dict': model.state_dict(),
           'model_optimizer_state_dict': model_optimizer.state_dict(),
           'epoch_loss': epoch_loss,
        }, os.path.join(dir_experiment, timestamp + "checkpoint_epoch_" + str(epoch) + ".th"))
        print("Saved Checkpoint.")

        if validate_during_training:
       
            valid_cost = validate_model(model, path_dataset_valid, display_test_results=True)

            #test_cost = validate_model(model, path_dataset_test, display_test_results=False)

            if not os.path.exists(os.path.join(dir_experiment, name_performance_validation)):
                 attribute_open = "w"
            else:
                 attribute_open = "a"
            # just a clone of below, but to also keep the "validation" loss (which is test loss, as we don't have a proper keep out validation set)
            with open(os.path.join(dir_experiment, name_performance_validation), attribute_open) as f:
                   f.write(str(valid_cost.detach().numpy()))
                   f.write("\n")
            

    # write progress to log file
    if not os.path.exists(os.path.join(dir_experiment, name_performance_epoch)):
        attribute_open = "w"
    else:
        attribute_open = "a"
    with open(os.path.join(dir_experiment, name_performance_epoch), attribute_open) as f:
        #if path_checkpoint is not None:
        #    f.write("Continuing Training...")
        f.write(str(epoch_loss))
        f.write("\n")

time_total_training = (time.time() - time_start_training) / 60
print("Finished Training Model within " + str(time_total_training) + " Minutes.")


########################################################################################
### EVALUATE TEST PERFORMANCE
########################################################################################

test_cost = validate_model(model, path_dataset_test, display_test_results=True)

#dict_experiment = {
#    "list_loss_training": list_loss_training,
#    "test_cost": test_cost,
#}
#np.save(timestamp + "dict_experiment.npy", dict_experiment)


## Plot Training Epoch Loss
##
# import glob
# import matplotlib.pyplot as plt
# import torch
#
# list_files = sorted(glob.glob("*.th"))
#
# losses=[]
#
# for f in list_files:
#     losses.append(torch.load(f)["epoch_loss"])
#
# plt.plot(range(1,len(losses)+1), losses); plt.scatter(range(1,len(losses)+1), losses); plt.show(); plt.clf()
