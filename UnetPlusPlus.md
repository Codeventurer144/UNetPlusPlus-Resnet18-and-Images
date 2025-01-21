```python
# IMPORTING THE NECESSARY
import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
import torchvision
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import DataLoader

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(512),
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_path = r'C:\Users\ENIIFEOLUWA S. OKE\Downloads\hymenoptera_data\hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=0)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(image_datasets, class_names)
```
```
{'train': Dataset ImageFolder
    Number of datapoints: 244
    Root location: C:\Users\ENIIFEOLUWA S. OKE\Downloads\hymenoptera_data\hymenoptera_data\train
    StandardTransform
Transform: Compose(
               Resize(size=512, interpolation=bilinear, max_size=None, antialias=True)
               RandomResizedCrop(size=(64, 64), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=bilinear, antialias=True)
               RandomHorizontalFlip(p=0.5)
               ToTensor()
               Normalize(mean=[0.5 0.5 0.5], std=[0.25 0.25 0.25])
           ), 'val': Dataset ImageFolder
    Number of datapoints: 153
    Root location: C:\Users\ENIIFEOLUWA S. OKE\Downloads\hymenoptera_data\hymenoptera_data\val
    StandardTransform
Transform: Compose(
               Resize(size=512, interpolation=bilinear, max_size=None, antialias=True)
               CenterCrop(size=(64, 64))
               ToTensor()
               Normalize(mean=[0.5 0.5 0.5], std=[0.25 0.25 0.25])
           )} ['ants', 'bees']
```

```python
sample_image, _ = image_datasets['train'][152]

# Converting the image back to PIL to confirm the size of a random image in the training dataset
pil_image = transforms.ToPILImage()(sample_image)
image_size = pil_image.size  # (width, height)
print(f"The size of the image is: {image_size}")
```

The size of the image is: (64, 64)

```python
for inputs, labels in image_datasets['train']:
    print(f"Input shape: {inputs.shape}, Label: {labels}")
    break
```

Input shape: torch.Size([3, 64, 64]), Label: 0

☝️ `This means that the labels are in scalar form and not dimensional representations of the images, I think that's fine because I am only doing binary classification.`


```python
class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(UNetPlusPlus, self).__init__()
        
        # Encoder layers (contracting path)
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # Pooling layer for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bridge (bottleneck layer)
        self.bridge = self.conv_block(512, 1024)
        
        # Decoder layers (expansive path with dense skip connections)
        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)
        
        # Final convolution layer
        self.final_conv = nn.Conv2d(64, 1,  kernel_size=1)
        
        # Global pooling to reduce spatial size
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer for binary classification
        self.fc = nn.Linear(64, num_classes)
        
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder forward pass
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Bridge forward pass
        bridge = self.bridge(self.pool(enc4))
        
        # Decoder with dense skip connections
        dec4 = self.decoder4(torch.cat([F.interpolate(bridge, scale_factor=2, mode='bilinear', align_corners=True), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True), enc3, enc4], dim=1))
        dec2 = self.decoder2(torch.cat([F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True), enc2, enc3, enc4], dim=1))
        dec1 = self.decoder1(torch.cat([F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True), enc1, enc2, enc3, enc4], dim=1))
        
        features = self.final_conv(dec1)
        
        pooled_features = self.global_pooling(features).view(features.size(0), -1)
        output = self.fc(pooled_features)
        output = self.sigmoid(output)
        
        return output
```

```python
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.unsqueeze(1).float()

                # Forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(torch.sigmoid(outputs).round() == labels.data)

                gc.collect() 

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model
```

```python
# NEXT THING TO DO IS TO INITIALIZE THE MODEL and TRAIN IT___FINGERS CROSSED
model = UNetPlusPlus(in_channels=3, num_classes=1)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss() #instead of Cross Entropy because i am doing a binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
```

```python
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=10)
```

```text
Epoch 0/9
----------
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[20], line 1
----> 1 model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=10)

Cell In[17], line 26, in train_model(model, criterion, optimizer, scheduler, num_epochs)
     23 labels = labels.unsqueeze(1).float()
     25 # Forward
---> 26 outputs = model(inputs)
     27 loss = criterion(outputs, labels)
     29 if phase == 'train':

File ~\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\nn\modules\module.py:1736, in Module._wrapped_call_impl(self, *args, **kwargs)
   1734     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1735 else:
-> 1736     return self._call_impl(*args, **kwargs)

File ~\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\nn\modules\module.py:1747, in Module._call_impl(self, *args, **kwargs)
   1742 # If we don't have any hooks, we want to skip the rest of the logic in
   1743 # this function, and just call forward.
   1744 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1745         or _global_backward_pre_hooks or _global_backward_hooks
   1746         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1747     return forward_call(*args, **kwargs)
   1749 result = None
   1750 called_always_called_hooks = set()

Cell In[16], line 57, in UNetPlusPlus.forward(self, x)
     55 # Decoder with dense skip connections
     56 dec4 = self.decoder4(torch.cat([F.interpolate(bridge, scale_factor=2, mode='bilinear', align_corners=True), enc4], dim=1))
---> 57 dec3 = self.decoder3(torch.cat([F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True), enc3, enc4], dim=1))
     58 dec2 = self.decoder2(torch.cat([F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True), enc2, enc3, enc4], dim=1))
     59 dec1 = self.decoder1(torch.cat([F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True), enc1, enc2, enc3, enc4], dim=1))

RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 16 but got size 8 for tensor number 2 in the list.
```



