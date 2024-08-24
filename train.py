import numpy as np
import pickle
import os
import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch.utils.data import DataLoader
import torch.optim as optim

class CNNModel(nn.Module):
  def __init__(self, numChannels, classes):
    super(CNNModel, self).__init__()
    self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,kernel_size=(5, 5))
    self.relu1 = ReLU()
    self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
    self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
    self.relu2 = ReLU()
    self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
    self.fc1 = Linear(in_features=800, out_features=500)
    self.relu3 = ReLU()
    self.fc2 = Linear(in_features=500, out_features=classes)
    self.logSoftmax = LogSoftmax(dim=1)

  def forward(self, x):
      x = self.conv1(x)
      x = self.relu1(x)
      x = self.maxpool1(x)
      x = self.conv2(x)
      x = self.relu2(x)
      x = self.maxpool2(x)
      x = flatten(x, 1)
      x = self.fc1(x)
      x = self.relu3(x)
      x = self.fc2(x)
      output = self.logSoftmax(x)
      return output
  
def load_model(model_path):
  model = CNNModel(1, 11)
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  checkpoint = torch.load(model_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']

  model.eval()
  print(f"Loaded model from epoch {epoch} with loss {loss}")
  return model, optimizer

def create_class_mapping(data_dir):
    filenames = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    class_names = set(os.path.splitext(f)[0] for f in filenames)

    class_mapping = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}
    return class_mapping

import numpy as np
import os
import matplotlib.pyplot as plt

def get_images_and_plot(data_dir, class_mapping, num_images=50):
    for class_name, class_idx in class_mapping.items():
        class_dir = os.path.join(data_dir, f"{class_name}.npy")
        images = np.load(class_dir)[:num_images]

        fig, axs = plt.subplots(1, num_images, figsize=(15, 3))
        for i, image in enumerate(images):
            axs[i].imshow(image.reshape(28,28), cmap='gray')
            axs[i].axis('off')
            axs[i].set_title(class_name)

        plt.show()

def predict_labels(model, image):
    image = image.reshape(28, 28) / 255
    image = image[np.newaxis, ...]  
    print(image)
    print(image.shape)
    model.eval()
    with torch.no_grad():
        image = torch.from_numpy(image).float().unsqueeze(1)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.tolist()
