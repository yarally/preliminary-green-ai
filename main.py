import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.io import imread
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.nn import *
from torch.optim import Adam
from tqdm import tqdm


class SimpleCNN(Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def load_mnist(problem, scope):
    logging.info('Loading data')
    labels_path = f'Problems/{problem}/{scope}-labels'
    images_path = f'Problems/{problem}/{scope}-images'

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as imgpath:
        images = []
        for image in np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28):
            img = image / 255.0
            img = img.astype('float32')
            images.append(img)

    return np.array(images), labels


def load_data(problem, scope):
    logging.info('Loading labels')
    data = pd.read_csv(f'Problems/{problem}/{scope}.csv')
    logging.info('Loading images')
    images = []
    for img_name in tqdm(data['id']):
        image_path = f'Problems/{problem}/{scope}/{str(img_name)}.png'
        img = imread(image_path, as_gray=True)
        img /= 255.0
        img = img.astype('float32')
        images.append(img)
    if 'label' in data:
        return np.array(images), data['label'].values
    return np.array(images)


def np_array_to_torch_tensor(x, y=None):
    dim1, dim2, dim3 = x.shape
    x = x.reshape(dim1, 1, dim2, dim3)
    x = torch.from_numpy(x)
    if y is not None:
        y = y.astype(int)
        y = torch.from_numpy(y)
        return x, y
    return x


def create_model(net):
    logging.info('Constructing CNN')
    model = net()
    optimizer = Adam(model.parameters(), lr=0.007)
    criterion = CrossEntropyLoss()
    if torch.cuda.is_available():
        logging.info('Using GPU')
        model = model.cuda()
        criterion = criterion.cuda()
    return model, optimizer, criterion


def train(net, epochs, train_x, train_y, val_x, val_y):
    logging.info('Training the model')
    model, optimizer, criterion = create_model(net)
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(epochs)):
        model.train()
        x_train, y_train = Variable(train_x), Variable(train_y)
        x_val, y_val = Variable(val_x), Variable(val_y)
        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            x_val = x_val.cuda()
            y_val = y_val.cuda()

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        # prediction for training and validation set
        output_train = model(x_train)
        output_val = model(x_val)

        # computing the training and validation loss
        loss_train = criterion(output_train, y_train)
        loss_val = criterion(output_val, y_val)
        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.show()
    return model


def test(model, test_x, test_y):
    logging.info('Generating predictions and calculating accuracy')
    with torch.no_grad():
        output = model(test_x.cuda())

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    return accuracy_score(test_y, predictions)


def predict(model, test_x, predictions):
    logging.info('Generating predictions')
    with torch.no_grad():
        output = model(test_x.cuda())

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions['label'] = np.argmax(prob, axis=1)
    return predictions


def run(problem, data_loader):
    # Prepare Data
    train_x, train_y = data_loader(problem, 'train')
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)
    train_x, train_y = np_array_to_torch_tensor(train_x, train_y)
    val_x, val_y = np_array_to_torch_tensor(val_x, val_y)

    # Construct and Train Model
    model = train(SimpleCNN, 100, train_x, train_y, val_x, val_y)

    # Accuracy on Test Set
    test_x, test_y = data_loader(problem, 'test')
    test_x = np_array_to_torch_tensor(test_x)
    accuracy = test(model, test_x, test_y)
    logging.info(f'{accuracy=}')
    # predictions_template = pd.read_csv(f'Problems/{problem}/prediction.csv')
    # predictions = predict(model, test_x, predictions_template)
    # predictions.to_csv('predictions.csv', index=False)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%I:%M:%S')
    torch.manual_seed(42)
    # run(problem='FashionMNIST', data_loader=load_mnist)
