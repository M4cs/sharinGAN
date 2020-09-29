import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable

#load data from google drive
from google.colab import drive
drive.mount('/content/gdrive')

#unzip the data
!unrar x "/content/gdrive/My Drive/sharingans.rar" "/content/gdrive/My Drive/sharingans/"

#sharingan images were resized to 128x128
#so the following shape is similar to [color channels, height, width]
img_shape = (3,128,128)

#Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(in_features = 100, out_features = 128)
        self.fc2 = nn.Linear(in_features = 128, out_features = 512)
        self.fc3 = nn.Linear(in_features = 512, out_features = 1024)
        self.fc4 = nn.Linear(in_features = 1024, out_features = 128*128*3)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        
    def forward(self, t):
        t = F.leaky_relu(self.fc1(t),0.2)
        t = F.leaky_relu(self.bn2(self.fc2(t)),0.2)
        t = F.leaky_relu(self.bn3(self.fc3(t)),0.2)
        t = torch.tanh(self.fc4(t))
        
        return t.view(t.shape[0], *img_shape)#changes the image to this shape

#Discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_features = 128*128*3, out_features = 512)
        self.fc2 = nn.Linear(in_features = 512, out_features = 256)
        self.fc3 = nn.Linear(in_features = 256, out_features = 128)
        self.fc4 = nn.Linear(in_features = 128, out_features = 1)
        
    def forward(self, t):
        t = t.view(t.size(0),-1)
        t = F.leaky_relu(self.fc1(t), 0.2)
        t = F.leaky_relu(self.fc2(t), 0.2)
        t = F.leaky_relu(self.fc3(t), 0.2)
        t = torch.sigmoid(self.fc4(t))
        return t

loss = torch.nn.BCELoss() #binary cross entropy loss function

#create the instances of the classes
generator = Generator()
discriminator = Discriminator()

#Loading the data
#I took the following load_images() function from StackOverflow
def load_images(image_size=128, batch_size=32, root="/content/gdrive/My Drive/sharingans/"):
    transform = transforms.Compose([
        #transforms.Resize(100),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.ImageFolder(root=root, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader

dataset = load_images()

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    loss_func.cuda()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas = (0.444, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas = (0.444, 0.999))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

#Training Loop
epochs = 3000
for epoch in range(epochs):
    for i, (images, labels) in enumerate(dataset):
        
        #ones is passed when the data is coming from original dataset
        #zeros is passed when the data is coming from generator
        ones = Tensor(images.size(0), 1).fill_(1.0)
        zeros = Tensor(images.size(0),1).fill_(0.0)
        
        real_images = images.cuda()
        
        optimizer_G.zero_grad()
        
        #following is the input to the generator
        #we create tensor with random noise of size 100
        gen_input = Tensor(np.random.normal(0,1,(images.shape[0],100)))
        #we then pass it to generator()
        gen = generator(gen_input) #this returns a image
        
        #now calculate the loss wrt to discriminator output
        g_loss = loss(discriminator(gen), ones)
        
        #backpropagation
        g_loss.backward()
        #update weights
        optimizer_G.step()
        
        #above was for generator network
        
        #now for the discriminator network
        optimizer_D.zero_grad()
        
        #calculate the real loss
        real_loss = loss(discriminator(real_images), ones)
        #calculate the fake loss from the generated image
        fake_loss = loss(discriminator(gen.detach()),zeros)
        #average out the losses
        d_loss = (real_loss + fake_loss)/2
        
        #backpropagation
        d_loss.backward()
        #update weights
        optimizer_D.step()
        
        if i%100 == 0:
            print("[EPOCH %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"%(epoch, epochs, i, len(dataset), d_loss.item(), g_loss.item()))

        total_batch = epoch * len(dataset) + i
        if total_batch%20 == 0:
            save_image(gen.data[:5], '/content/gdrive/My Drive/sgan/%d.png' % total_batch, nrow=5)


#testing part
noise = Tensor(np.random.normal(0,1,(images.shape[0],100)))

pred = generator(noise)

save_image(pred.data[:25], '/content/gdrive/My Drive/sgan/final.png', nrow=5)

#save models
torch.save(discriminator.state_dict(), '/content/gdrive/My Drive/discriminator.ckpt')
torch.save(generator.state_dict(), '/content/gdrive/My Drive/generator.ckpt')
