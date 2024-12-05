import torch 
import torch.nn
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class custom_dataset(Dataset):
    def __init__(self, mode = "train", root = "datasets", transforms = None):
        super().__init__()
        self.mode = mode
        self.root = root
        self.transforms = transforms
        
        #select split
        self.folder = os.path.join(self.root, self.mode)
        
        #initialize lists
        self.image_list = []
        self.label_list = []
        
        #save class lists
        self.class_list = os.listdir(self.folder)
        self.class_list.sort()
        
        for class_id in range(len(self.class_list)):
            for image in os.listdir(os.path.join(self.folder, self.class_list[class_id])):
                self.image_list.append(os.path.join(self.folder, self.class_list[class_id], image))
                label = np.zeros(len(self.class_list))
                label[class_id] = 1.0
                self.label_list.append(label)

    def __getitem__(self, index):
        
        image_name = self.image_list[index]
        label = self.label_list[index]
        
        image = Image.open(image_name)
        if(self.transforms):
            image = self.transforms(image)
        
        label = torch.tensor(label)
        
        return image, label
                
    def __len__(self):
        return len(self.image_list)       


# # Visualize a sample image
# dataset = custom_dataset(mode="train", root="datasets", transforms=transform)
# image, label = dataset[19]

# # Denormalize the image for visualization
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# image = image.numpy()
# image = image.transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
# image = (image * std) + mean      # Denormalize
# image = np.clip(image, 0, 1)      # Clip to valid range [0, 1]

# # Plot the image
# plt.imshow(image)
# plt.title(f"Label: {label}")
# plt.axis('off')
# plt.show()
