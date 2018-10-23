import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import time
import util
class AlexNet(nn.Module):

    def __init__(self, num_classes=3):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 8)
        x = self.classifier(x)
        return x

def train_alexNet(image_train_path,model,optimizer,loss_func,batch_size,epochs,length,width):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        image_train_path,
        transforms.Compose([
            transforms.Scale([length,width]),
            transforms.ToTensor(),
            normalize
        ]))
    print ("labels is ",train_dataset.classes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=2)
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        global_epoch_loss = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            # print inputs.shape
            # print labels
            inputs, labels = Variable(inputs), Variable(labels)
            output = model(inputs)               # cnn output
            # print (output)
            loss = loss_func(output, labels)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
            global_epoch_loss += loss.data[0]
            if i % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(inputs), len(train_loader.dataset), 100.* i / len(train_loader), loss.data[0]))
        loss = global_epoch_loss / len(train_loader.dataset)
        loss = "{:.4f}".format(loss)
    # return global_epoch_loss / len(train_loader.dataset)
    end_time = time.time()
    training_time = end_time-start_time
    training_time = "{:.4f} s".format(training_time)
    return loss,training_time

def valide_alexNet(image_validation_path,model,batch_size,length,width):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    validation_dataset = datasets.ImageFolder(
        image_validation_path,
        transforms.Compose([
            transforms.Scale([length,width]),
            transforms.ToTensor(),
            normalize
        ]))
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,shuffle=True, num_workers=2)

    model.eval()
    correct = 0
    pred_y = np.array([])
    test_y = np.array([])
    for i, data in enumerate(validation_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        output = model(inputs)
        pred = output.max(1, keepdim=True)[1]
        pred_y = np.append(pred_y,pred)
        test_y = np.append(test_y,labels)

        correct += pred.eq(labels.data.view_as(pred)).sum()
    # print correct.numpy()
    # print len(test_loader.dataset)
    # print float(correct) /len(test_loader.dataset)
    # print (metrics.classification_report(test_y, pred_y))
    accuracy,precision,recall,f1 = util.evaluate(test_y, pred_y)
    return float(correct) /len(validation_loader.dataset),model,accuracy,precision,recall,f1


if __name__ == "__main__":
    model = AlexNet()
    print(model)
    image_train_path = 'image/train'
    image_validation_path = 'image/validation'
    batch_size = 5
    length = 224
    width = 299
    epochs = 2
    lr = 0.1
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    train_alexNet(image_train_path,model,optimizer,loss_func,batch_size,epochs,length,width)
    valide_alexNet(image_validation_path,model,batch_size,length,width)
