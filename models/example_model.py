import torch
import torch.nn as nn
import torchvision.models as model
import torch.nn.functional as F


class ExModel(nn.Module):
    
    def __init__(self):
        super().__init__()
                
        # Resnet18
        # self.resnet18 = model.resnet18(pretrained = True)
        # self.resnet18 = torch.nn.Sequential(*(list(self.resnet18.children())[:-1]))    

        # for param in self.resnet18.parameters():
        #     param.requires_grad = False

        # self.classifier = torch.nn.Linear(512, 2)

        
        # Vgg16
        self.vgg16 = model.vgg16(pretrained = True)
        self.vgg16.classifier = torch.nn.Sequential(*(list(self.vgg16.classifier.children())[:-1]))

        for param in self.vgg16.parameters():
            param.requires_grad = False

        self.classifier = torch.nn.Linear(4096, 2)

    def forward(self, image):
        # Resnet18
        # resnet_pred = self.resnet18(image).squeeze()
        # out = self.classifier(resnet_pred)

        # Vgg16
        # vgg_pred = self.vgg16(image).squeeze()
        # out = self.classifier(vgg_pred)

        image = self.vgg16(image)
        # Flatten the output
        image = image.view(image.size(0), -1)
        # Pass through the new classifier
        out = self.classifier(image)

        return out

    