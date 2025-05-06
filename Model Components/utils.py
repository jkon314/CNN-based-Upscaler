import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def get_activation_function(activation_function):
    if activation_function == "relu":
        return nn.ReLU()
    elif activation_function == "tanh":
        return nn.Tanh()
    elif activation_function == "softmax":
        return nn.Softmax()
    else:
        return nn.ReLU()


def streamFrames(chunkSize=1000,startFrame=1,vidName1080='CodGP',vidName720 ='CODGP') -> tuple :

    """
    Parameters:
    chunkSize (int): 


    """

    fPath = '../Data/Frames/'
    
    targetFrames = []

    inFrames =  []

    for i in range(startFrame,startFrame+chunkSize):
        

        print(f'Processing frame {i}/{chunkSize}',end='\r')

        tar = cv2.imread(f'{fPath}{vidName1080}_frame_{i}.jpg',cv2.IMREAD_COLOR)
        source = cv2.imread(f'{fPath}{vidName720}_720_frame_{i}.jpg',cv2.IMREAD_COLOR)
        
        #slice up higher res image
        tar_slices = sixty_four_chunk_image(tar)

        for slice in tar_slices:
            targetFrames.append(slice.reshape((slice.shape[0]*slice.shape[1],slice.shape[2])))

        
        source_slices = sixty_four_chunk_image(source)
        #reshape to work with Pytorch Expected Order (Channels, Height, Width)

        for slice in source_slices:
            inFrames.append(slice.reshape((slice.shape[2],slice.shape[0],slice.shape[1])))


    return (np.array(targetFrames),np.array(inFrames))

def criterion(highResTarget , decoder_output): 
    mse_loss = nn.MSELoss()
    
    b,g,r = decoder_output

    Loss1 = mse_loss(highResTarget[:,:,0], b)
    Loss2 = mse_loss(highResTarget[:,:,1], g)
    Loss3 = mse_loss(highResTarget[:,:,2], r)
    return Loss1 + Loss2 + Loss3


def train(model, dataloader, optimizer, criterion, device='cpu'):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", ascii=True)

    for batch_idx, data in enumerate(progress_bar):

        target, source = data
        target, source = target.to(device), source.to(device)

        optimizer.zero_grad()

        source = source.float()
        target = target.float()

        decoder_output = model(source)

        loss = criterion(target,decoder_output)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss


def evaluate(model, dataloader, criterion, device='cpu'):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", ascii=True)

        for batch_idx, data in enumerate(progress_bar):
            target, source = data
            target, source = target.to(device), source.to(device) 

            target = target.float()
            source = source.float()

            decoder_output = model(source)
            loss = criterion(target,decoder_output) #edited to unpack multiple returns
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss



def plot_curves(train_loss_history, valid_loss_history, filename):
    '''
    Plot learning curves with matplotlib. Training loss and validation loss are plotted in the same figure.
    :param train_loss_history: training loss history of epochs
    :param valid_loss_history: validation loss history of epochs
    :param filename: filename for saving the plot
    :return: None, save plot in the current directory
    '''
    epochs = range(len(train_loss_history))
    # plt.plot(epochs, train_loss_history, label='Train Loss')
    # plt.plot(epochs, valid_loss_history, label='Validation Loss')

    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Training and Validation Loss Curves - ' + filename)
    # plt.savefig(filename + '.png')
    # plt.show()

def four_chunk_image(img: np.array) -> np.array:
    width = img.shape[1]
    height = img.shape[0]
    

    top_left = img[:int(height/2),:int(width/2),:]
    top_right = img[:int(height/2),int(width/2):,:]
    bottom_left = img[int(height/2):,:int(width/2),:]
    bottom_right = img[int(height/2):,int(width/2):,:]

    return [top_left,top_right,bottom_left,bottom_right]
def sixty_four_chunk_image(img: np.array) -> np.array:
    '''
    Create 64 building block images from an original image to make size manageable
    :param img: np array containing the original image
  
    :return: np.array containing 64 image chunks from original image
    '''
    eigth_width = int(img.shape[1]/8)
    eigth_height = int(img.shape[0]/8)
    chunkedImages = []
    for i in range(8):

        for j in range(8):
            chunkedImages.append(img[i*eigth_height:(i+1)*eigth_height,j*eigth_width:(j+1)*eigth_width,:])

    return np.array(chunkedImages)

def reconstruct_image(fragments: tuple) -> np.array:


    upscaled_chunk_h = 135 
    upscaled_chunk_w = 240
    sqrt_chunks = 8

    #step 1 create image frame to place image into

    frame = np.zeros((1080,1920,3))


    #step 2 unpack all of the image fragments and place them into the frame using slicing
    

    #64 fragments, 8x8 grid, 3 channel 
    #output comes out as a tuple of tensors
    blue_channels = fragments[0].cpu().detach().numpy()
    green_channels = fragments[1].cpu().detach().numpy()
    red_channels = fragments[2].cpu().detach().numpy()
    
    num_fragments = blue_channels.shape[0]
   #reshape the channels to match the dimensions of upscaled image
    blue_channels = blue_channels.reshape((num_fragments,upscaled_chunk_h,upscaled_chunk_w))
    green_channels = green_channels.reshape((num_fragments,upscaled_chunk_h,upscaled_chunk_w))
    red_channels = red_channels.reshape((num_fragments,upscaled_chunk_h,upscaled_chunk_w))
    for i in range(0,sqrt_chunks):

        for j in range(0,sqrt_chunks):

            #fill blue channel
            frame[upscaled_chunk_h*i:upscaled_chunk_h*(i+1),upscaled_chunk_w*j:upscaled_chunk_w*(j+1),0] =  blue_channels[i*sqrt_chunks+j]
            #fill green channel
            frame[upscaled_chunk_h*i:upscaled_chunk_h*(i+1),upscaled_chunk_w*j:upscaled_chunk_w*(j+1),1] = green_channels[i*sqrt_chunks+j]
            #fill red channel
            frame[upscaled_chunk_h*i:upscaled_chunk_h*(i+1),upscaled_chunk_w*j:upscaled_chunk_w*(j+1),2] = red_channels[i*sqrt_chunks+j]


    return frame

def convert_to_input(img: np.array) -> np.array:

    #reshape array to be suited to input for 
    inFrames = []
    #cut image into VRAM manageable pieces
    chunks = sixty_four_chunk_image(img)
    for slice in chunks:
        #create Conv 2d suited shape and append to input array
        inFrames.append(slice.reshape((slice.shape[2],slice.shape[0],slice.shape[1])))

    return np.array(inFrames)