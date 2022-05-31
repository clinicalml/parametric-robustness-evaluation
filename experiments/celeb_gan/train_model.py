from torch.utils.data import DataLoader
from experiments.celeb_gan.preprocessing import CelebADataset
import torchvision.models as models
from tqdm import tqdm
from torch import nn
import time
import os
import argparse
import torch
import copy

IMG_PATH = 'data/celeb_gan/train_dist/images'
META_PATH ='data/celeb_gan/train_dist'
MODEL_PATH = 'experiments/celeb_gan/models/resnet_finetuned.pt'
# PRED_PATH = 'experiments/celeb_gan/models'

# Setups
batch_size = 256
device = "cuda"

# Number of classes in the dataset
num_classes = 2
feature_extract = True

parser = argparse.ArgumentParser()
parser.add_argument('--folder_name', type=str, default='default')
parser.add_argument('--num_epochs', type=int, default=25)


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, metadata in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                labels = labels.squeeze()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    args = parser.parse_args()
    variables_path = os.path.join("experiments/celeb_gan/variables/", args.folder_name)
    if not os.path.exists(variables_path):
        os.makedirs(variables_path)

    num_epochs = args.num_epochs

    model = models.wide_resnet50_2(pretrained=True)
    set_parameter_requires_grad(model, feature_extracting=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)


    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:

                print("\t",name)



    dataset = CelebADataset(meta_path=META_PATH, img_path=IMG_PATH, target_name='Male', train_frac = 0.6, val_frac=0.1)
    splits = ['train', 'val', 'test']
    dataloaders = {split: DataLoader(dataset=ds, 
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=16) for split, ds in dataset.get_splits(splits).items()}
    
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model, val_acc_history = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs)

    # Save the model checkpoint
    torch.save(model.state_dict(), MODEL_PATH)

    # Load data
    test_dataloader = dataloaders['test']
    preds       = torch.empty((0, 1)).to(device)

    # Iterate over data
    with torch.no_grad():
        model.eval()
        for inputs, labels, metadata in tqdm(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Compute predictions
            labels = labels.squeeze()
            outputs = model(inputs)
            preds = torch.cat((preds, outputs.argmax(dim=1).view(-1, 1)), dim=0)

        torch.save(preds, os.path.join(variables_path, 'preds.pt'))

    # Save metadata and targets
    metas = dataset.attrs_df.loc[dataset.get_splits(['test'])['test'].indices]
    targets = torch.tensor(dataset.y_array[dataset.get_splits(['test'])['test'].indices])
    metas.to_csv(os.path.join(variables_path, 'metas.csv'), index=False)
    torch.save(targets, os.path.join(variables_path, 'targets.pt'))

    acc = (1.0*(preds == targets.to(device).view(-1, 1))).float()
    print("Test accuracy:", acc.mean().item())
