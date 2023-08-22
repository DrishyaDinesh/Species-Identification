
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy
from PIL import Image
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Module
from sklearn.preprocessing import LabelEncoder

def predict(image_path, model_path, label_encoder_file):

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # define model class
  num_classes = 20

  class classifier(Module):
      def __init__(self):
          super(classifier, self).__init__()

          self.cnn_layers = Sequential(
              # Defining a 2D convolution layer
              Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
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
              Linear(12544, 64),
              Linear(64, num_classes)
          )

      # defining the forward pass
      def forward(self, x):
          x = self.cnn_layers(x)
          x = x.view(x.size(0), -1)
          x = self.linear_layers(x)
          return x

  model = classifier()

  # load the model weights
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.eval()

  # define the image transform
  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
  
  # process the image
  image1 = Image.open(image_path)
  image = transform(image1)
  image = image.unsqueeze(0)

  # make predictions
  image = image.to(device)
  # print(image.shape)
  output = model(image)

  # get probability scores
  prob = F.softmax(output, dim=1)

  # get the top 3 predictions
  top_p, top_class = prob.topk(3, dim = 1)
  top_p, top_class = top_p[0].tolist(), top_class[0].tolist()

  for i in range(len(top_p)):
    top_p[i] = round(top_p[i], 3)

  # import the label encoder and inverse transform the labels
  encoder = LabelEncoder()
  encoder.classes_ = numpy.load(label_encoder_file, allow_pickle=True)

  top_cats = []
  for i in top_class:
    top_cats.append(encoder.inverse_transform([i])[0])

  return top_p, top_cats
