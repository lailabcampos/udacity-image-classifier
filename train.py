import torch
import utils
import model
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('data_directory', help='Directory of the files')
parser.add_argument('--save_dir', action='store',
                    dest='save_dir',
                    help='Directory to save the model')
parser.add_argument('--arch', action='store',
                    dest='arch',
                    default='vgg16',
                    help='Architecture of the model')
parser.add_argument('--learning_rate', action='store',
                    dest='learning_rate',
                    default=0.0005,
                    help='Learning rate of the model')
parser.add_argument('--hidden_units', action='append',
                    dest='hidden_units',
                    default=[2048, 512],
                    help='Hidden units of the model')
parser.add_argument('--ephocs', action='store',
                    dest='epochs',
                    default=8,
                    help='Epochs to train the model')
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Set training to gpu')

results = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataloaders, dataset_sizes = utils.preprocessing_images(results.data_dir)
model, criterion, optimizer = model.model(hidden_layers=results.hidden_units, 
                                          learning_rate=results.learning_rate, 
                                          arch=results.arch, 
                                          device=device)

# Training new model
epochs = results.epochs
steps = 0
running_loss = 0

train_losses, valid_losses = [], []

for e in range(epochs):
    for images, targets in trainloader:
        steps += 1
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()        
    else:
        valid_loss = 0
        accuracy = 0
        model.eval()
        
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                log_ps = model.forward(images)

                valid_loss += criterion(log_ps, labels).item()
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        print(f"Epoch {e+1}/{epochs} (steps: {steps}).. "
              f"Train loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation loss: {valid_loss/len(validloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validloader):.3f}.. ")
        
        train_losses.append(running_loss)
        valid_losses.append(valid_loss)
        running_loss = 0
        steps = 0
        model.train()
        
# Validation

test_loss = 0
accuracy = 0

with torch.no_grad():
    model.eval()
    for images, labels in testloader:
        log_ps = model(images)
        test_loss += criterion(log_ps, labels)
        
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))

print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

# Saving checkpoint

checkpoint = {'epochs': epochs,
              'train_losses': train_losses,
              'valid_losses': valid_losses,
              'layers': ,
              'optimizer_state_dict': optimizer.state_dict(),
              'model_state_dict': model.state_dict()}

torch.save(checkpoint, 'model_checkpoint.pth')