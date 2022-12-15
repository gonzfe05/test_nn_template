from torch import nn


# https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
class CNN(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential()
        self.out = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.model(x)
        # [batch_size, 32 * 7 * 7]
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


class SiameseNetwork(nn.Module):
    def __init__(self, num_classes: int):
        super(SiameseNetwork, self).__init__()
        assert num_classes == 2
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(512, 2),
        )

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2


# class SiameseNetwork(nn.Module):
#     def __init__(self, num_classes: int):
#         super(SiameseNetwork, self).__init__()
#         assert num_classes == 2
#         # Setting up the Sequential of CNN Layers
#         self.cnn1 = nn.Sequential(
#             nn.Conv2d(1, 96, kernel_size=11,stride=1),
#             nn.ReLU(inplace=True),
#             nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
#             nn.MaxPool2d(3, stride=2),

#             nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
#             nn.ReLU(inplace=True),
#             nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
#             nn.MaxPool2d(3, stride=2),
#             nn.Dropout2d(p=0.3),

#             nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, stride=2),
#             nn.Dropout2d(p=0.3),
#         )
#         # Defining the fully connected layers
#         self.fc1 = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.5),

#             nn.Linear(128, 64),
#             nn.ReLU(inplace=True),

#             nn.Linear(64,2))

#     def forward_once(self, x):
#         # Forward pass
#         output = self.cnn1(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc1(output)
#         return output

#     def forward(self, input1, input2):
#         # forward pass of input 1
#         output1 = self.forward_once(input1)
#         # forward pass of input 2
#         output2 = self.forward_once(input2)
#         return output1, output2
