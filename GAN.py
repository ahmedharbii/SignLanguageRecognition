#Generative Adversarial Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


#Hyperparameters:
batch_size = 128
image_size = 64
num_epochs = 5
lr = 0.0002
workers = 2
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
#Define the Generator to generate images:
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            #Input is Z, going into a convolution:
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #State size. (512x4x4)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #State size. (256x8x8)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            #State size. (128x16x16)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #State size. (64x32x32)
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            #State size. (3x64x64)

        )
    def forward(self, input):
        return self.main(input)

#Define the Discriminator to classify images:
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            #Input is 3x64x64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #State size. (64x32x32)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            #State size. (128x16x16)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            #State size. (256x8x8)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            #State size. (512x4x4)
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)

#Initialize the Generator and Discriminator:
netG = Generator().to(device)
netD = Discriminator().to(device)

#Handle multi-gpu if desired:
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))

#Initialize the weights:
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
netG.apply(weights_init)
netD.apply(weights_init)

#Print the Generator model:
print(netG)

#Print the Discriminator model:
print(netD)

#Initialize the BCELoss function:
criterion = nn.BCELoss()

#Create the batch of latent vectors that we will use to visualize the progression of the Generator:
fixed_noise = torch.randn(64, 100, 1, 1, device=device)

#Setup Adam optimizers for both G and D:
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

#Training Loop:
#Lists to keep track of progress:
def train_model(dataloader, device, num_epochs=5, lr=0.0002):
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    #For each epoch:
    for epoch in range(num_epochs):
        #For each batch in the dataloader:
        for i, data in enumerate(dataloader, 0):
            #Update D network:
            #Train with all-real batch:
            netD.zero_grad()
            #Format batch:
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1, device=device)
            #Forward pass real batch through D:
            output = netD(real_cpu).view(-1)
            #Calculate loss on all-real batch:
            errD_real = criterion(output, label)
            #Calculate gradients for D in backward pass:
            errD_real.backward()
            D_x = output.mean().item()

            #Train with all-fake batch:
            #Generate batch of latent vectors:
            noise = torch.randn(b_size, 100, 1, 1, device=device)
            #Generate fake image batch with G:
            fake = netG(noise)
            label.fill_(0)
            #Classify all fake batch with D:
            output = netD(fake.detach()).view(-1)
            #Calculate D's loss on the all-fake batch:
            errD_fake = criterion(output, label)
            #Calculate the gradients for this batch:
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            #Add the gradients from the all-real and all-fake batches:
            errD = errD_real + errD_fake
            #Update D:
            optimizerD.step()

            #Update G network:
            netG.zero_grad()
            label.fill_(1)  # fake labels are real for generator cost
            #Since we just updated D, perform another forward pass of all-fake batch through D:
            output = netD(fake).view(-1)
            #Calculate G's loss based on this output:
            errG = criterion(output, label)
            #Calculate gradients for G:
            errG.backward()
            D_G_z2 = output.mean().item()
            #Update G:
            optimizerG.step()

            #Output training stats:
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            #Save Losses for plotting later:
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            #Check how the Generator is doing by saving G's output on fixed_noise:
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    return img_list, G_losses, D_losses

#Data loader:
batch_size = 128
PATH_DATASET = os.path.expanduser('~') + '/Datasets/SignLanguageLetters/train'
dataset = dset.ImageFolder(root=PATH_DATASET, transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
]))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


img_list, G_losses, D_losses = train_model(dataloader, device, num_epochs=5, lr=0.0002)

#Plot the training losses:
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#Plot the Generator output:
#Grab a batch of real images from the dataloader:
real_batch = next(iter(dataloader))

#Plot the real images:
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

#Plot the fake images from the last epoch:
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()

#Animation showing the improvements of the Generator:
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)


#Save the model:
torch.save(netG.state_dict(), 'netG.pth')
torch.save(netD.state_dict(), 'netD.pth')

#Load the model:
netG = Generator(ngpu).to(device)
netG.load_state_dict(torch.load('netG.pth'))

#Generate a batch of images:
noise = torch.randn(64, 100, 1, 1, device=device)
fake = netG(noise)

#Plot the fake images:
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(fake.detach().cpu(), padding=2, normalize=True),(1,2,0)))
plt.show()



