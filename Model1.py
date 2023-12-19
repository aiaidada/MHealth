import os
import shutil
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image

# Define the path to the main folder containing the subfolders
main_folder_path = "D:\Dars\images_all/images"

# Define the root folder for the datasets
root_dataset_folder = "/content2/dataset"

# Iterate through folders 1 to 7
for x in range(1, 11):
    # Define folder names for good and bad datasets
    good_folder_name = f"{x}_good"
    bad_folder_name = f"{x}_bad"

    # Create dataset folders
    dataset_folder_path = os.path.join(root_dataset_folder, f"dataset_{x}")
    os.makedirs(dataset_folder_path, exist_ok=True)

    # Process good files with label 0
    good_folder_path = os.path.join(main_folder_path, good_folder_name)
    good_dataset_path = os.path.join(dataset_folder_path, "good")
    os.makedirs(good_dataset_path, exist_ok=True)
    for file_name in os.listdir(good_folder_path):
        file_path = os.path.join(good_folder_path, file_name)
        shutil.copy(file_path, os.path.join(good_dataset_path, file_name))

    # Process bad files with label 1
    bad_folder_path = os.path.join(main_folder_path, bad_folder_name)
    bad_dataset_path = os.path.join(dataset_folder_path, "bad")
    os.makedirs(bad_dataset_path, exist_ok=True)
    for file_name in os.listdir(bad_folder_path):
        file_path = os.path.join(bad_folder_path, file_name)
        shutil.copy(file_path, os.path.join(bad_dataset_path, file_name))


print("Datasets created successfully.")
import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['good', 'bad']
        self.file_paths, self.labels = self._load_file_paths_and_labels()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx])
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

    def _load_file_paths_and_labels(self):
        file_paths = []
        labels = []

        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_path):
                class_label = 0 if class_name == 'good' else 1
                for filename in os.listdir(class_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        file_paths.append(os.path.join(class_path, filename))
                        labels.append(class_label)

        return file_paths, labels

import torch.optim as optim
import time

def train(epochs , net , train_loader ):

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  start_time = time.time()

  for epoch in range(epochs):  # loop over the dataset multiple times

    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs.unsqueeze(0), labels)
        loss.backward()
        optimizer.step()

  end_time = time.time()
  training_time = end_time - start_time
  print(f'Training Time: {training_time:.2f} seconds')
  print('Finished Training')
  return training_time

def evaluate_model(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images, labels
            outputs = model(images)
            total_correct = total_correct+1 if outputs.argmax() == labels else total_correct
            total_samples +=1

    accuracy = total_correct / total_samples
    return accuracy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(6032, 2500)
        self.fc2 = nn.Linear(2500, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



transforms = transforms.Compose([transforms.Resize((64, 128)), transforms.ToTensor()])
train_times = [0]*10
train_acc= [0]*10
test_acc = [0]*10

models = []
print(len(models))
for i in range(1 , 11):
  net = Net()
  dataset = CustomDataset(root_dir= f'/content2/dataset/dataset_{i}', transform= transforms)
  for j in range(10):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    time_train = train(epochs=50 , net= net, train_loader= train_loader )
    train_times[i-1] += time_train
    train_accuracy = evaluate_model(net, train_loader)
    print(f'Training Accuracy: {train_accuracy:.4f}' , i)
    train_acc[i-1]+= train_accuracy
    # Evaluate the model on the test set
    test_accuracy = evaluate_model(net, val_loader)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    test_acc[i-1]+= test_accuracy
    models.append(net)
print(train_acc)
print(test_acc)
print(train_times)
torch.save(models , 'Model_1')