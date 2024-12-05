from datasets.dataset_retrieval import custom_dataset
import torch
from torchmetrics import F1Score
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from models.example_model import ExModel

def load_model(checkpoint_path, device):
    model = ExModel()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

def test(model, test_loader, device):
    f1score = 0
    f1 = F1Score(num_classes = 2, task = 'binary')
    data_iterator = enumerate(test_loader)  # take batches

    f1_list = []
    f1t_list = []

    classwise_accuracy = {
        "0": 0,
        "1": 0,
        "true_0" : 0,
        "true_1" : 0
    }

    with torch.no_grad():

        for _, batch in data_iterator:
            image, label = batch
            image = image.to(device)
            label = label.to(device)
            pred = model(image)

            pred = pred.softmax(dim = 1)
            
            f1_list.extend(torch.argmax(pred, dim =1).tolist())
            f1t_list.extend(torch.argmax(label, dim =1).tolist())
    
    f1score = f1(torch.tensor(f1_list), torch.tensor(f1t_list))

    for prediction, label in zip(f1_list, f1t_list):
        if label == 0:
            classwise_accuracy["0"] += 1
            if prediction == 0:
                classwise_accuracy["true_0"] += 1
        else:
            classwise_accuracy["1"] += 1
            if prediction == 1:
                classwise_accuracy["true_1"] += 1
        

    return f1score, classwise_accuracy

def main():
    device = "cuda"
    model_path = "checkpoints/vgg_adam.pth"
    model = load_model(model_path, device)
    
    tr = transforms.Compose([
        transforms.ToTensor(),

        transforms.Resize([224, 224]),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_data = custom_dataset("test", transforms = tr)

    test_loader = DataLoader(
        test_data,
        batch_size = 32,
        shuffle = False
    )

    f1_score, class_wise = test(model, test_loader, device)

    print(f"Accuracy F1 : {f1_score}\nClasswise 0(glass) : {class_wise["true_0"] / class_wise["0"]}\nClasswise 1(no_glass) : {class_wise["true_1"] / class_wise["1"]}")

if __name__ == "__main__":
    main()    
