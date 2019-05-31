import torch
from torch import nn, optim
from torchvision import models
from collections import OrderedDict

def model(hidden_layers=[2048, 512], learning_rate=0.0005, arch="vgg16", device="cpu"):
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, int(hidden_layters[0])),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.6)),
        ('fc2', nn.Linear(int(hidden_layters[0]), int(hidden_layters[1]))),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.6)),
        ('fc3', nn.Linear(int(hidden_layters[1]), 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    if device == "cuda" and torch.cuda.is_available(): model.to("cuda")
    else: model.to("cpu")

    return model, criterion, optimizer
        
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    loaded_model, _, _ = model(hidden_layers = checkpoint['layers'][1:3],
                         learning_rate = 0.0005,
                         arch = "vgg16",
                         device = "cpu")
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.class_to_idx = checkpoint['class_to_idx']
    return loaded_model