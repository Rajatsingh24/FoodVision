"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import torch
import torchvision
from torch import nn
import data_setup
import engine
import model_builder
import utils

# Setup hyperparameters 
NUM_EPOCHS = [7, 4, 3]  # [feature Extraction, Fine Tuning Part 1, Fine Tuning Part 2]
BATCH_SIZE = 32
LEARNING_RATE = [0.001, 0.0001, 0.00001]  # [feature Extraction, Fine Tuning Part 1, Fine Tuning Part 2]

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Create transforms
data_transform = torchvision.models.EfficientNet_B2_Weights.DEFAULT.transforms()
# ------------------------------------------ DataLoaders ----------------------------------------------------------------------#
# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    transform=data_transform,
    batch_size=BATCH_SIZE)
print("dataloaders created")

# ------------------------------------------ Model ----------------------------------------------------------------------------#
# Create model with help from model_builder.py
model = model_builder.model_build(device=device)
print("model created")

# ------------------------------------------ Feature Extraction ---------------------------------------------------------------#
# Setting all parameters to not-trainable
for params in model.parameters():
    params.requires_grad = False

# Changing Classification layer
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features=1408, out_features=len(class_names)))
# model.classifier

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE[0])

# Start training with help from engine.py
feature_extraction_results=engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS[0],
             device=device)
print(feature_extraction_results)

# ------------------------------------------ Fine Tuning Part 1 ---------------------------------------------------------------#
# Setting models upper layer un froze
for params in model.features[5:].parameters():
    params.requires_grad = True

for m in model.modules():  # Making the BatchNorm2d froze
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = False
        m.eval()

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE[1])

# Start training with help from engine.py
fine_tuning_p1_results=engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS[1],
             device=device)
print(fine_tuning_p1_results)

# ------------------------------------------ Fine Tuning Part 2 ---------------------------------------------------------------#
# Setting models upper layer un froze
for params in model.features.parameters():
    params.requires_grad = True

for m in model.modules():  # Making the BatchNorm2d froze
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = False
        m.eval()

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE[2])

# Start training with help from engine.py
fine_tuning_p2_results=engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS[2],
             device=device)
print(fine_tuning_p2_results)

# ------------------------------------------ Save model -----------------------------------------------------------------------#
# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="Image_Classification_EfficientNet_B2.pth")
