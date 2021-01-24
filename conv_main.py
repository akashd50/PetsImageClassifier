import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from ConvNet import ConvNet
from ImgProcessor import ImgProcessor
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

REBUILD_DATA = False
SHOULD_RETRAIN_NET = False
VAL_PCT = 0.1
BATCH_SIZE = 100
EPOCHS = 3
MODEL_PATH = "model/conv_net.pth"


def get_best_device():
    device = None
    print("Total CUDA Devices: ", torch.cuda.device_count())
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    return device


def load_training_data():
    # Load Training Data
    if REBUILD_DATA:
        training_data_processor = ImgProcessor()
        training_data_processor.make_training_data()

    training_data = np.load("training_data.npy", allow_pickle=True)
    # print(len(training_data))
    # plt.imshow(training_data[0][0], cmap="gray")
    # plt.show()
    return training_data


def separate_training_and_testing_data(training_data):
    # Normalize Training data from gray values between (0, 255) to 0.0 - 1.0
    x = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
    x = x / 255.0
    y = torch.Tensor([i[1] for i in training_data])

    # Initialize training vs testing suites
    val_size = int(len(x) * VAL_PCT)
    print("Test size: ", val_size)

    train_x = x[:-val_size]
    train_y = y[:-val_size]

    test_x = x[-val_size:]
    test_y = y[-val_size:]
    return train_x, train_y, test_x, test_y


def do_training(net, optimizer, loss_function, device, train_x, train_y):
    # Starting training
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_x), BATCH_SIZE)):
            batch_x = train_x[i:i + BATCH_SIZE].view(-1, 1, 50, 50).to(device)
            batch_y = train_y[i:i + BATCH_SIZE].to(device)

            net.zero_grad()
            # can use optimizer.zero_grad() if the optimizer is controlling all parameters
            outputs = net(batch_x)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
    print(loss)


def do_testing(net, device, test_x, test_y):
    # Final Testing
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_x))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_x[i].view(-1, 1, 50, 50).to(device))[0]
            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy: ", round(correct / total, 3))


def main():
    device = get_best_device()
    training_data = load_training_data()

    # Initialize Net Object, Optimizer and Loss function
    net = ConvNet().to(device)

    if not SHOULD_RETRAIN_NET:
        net.load_state_dict(torch.load(MODEL_PATH))

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    train_x, train_y, test_x, test_y = separate_training_and_testing_data(training_data)

    if SHOULD_RETRAIN_NET:
        do_training(net, optimizer, loss_function, device, train_x, train_y)
        torch.save(net.state_dict(), MODEL_PATH)

    do_testing(net, device, test_x, test_y)


if __name__ == '__main__':
    main()
