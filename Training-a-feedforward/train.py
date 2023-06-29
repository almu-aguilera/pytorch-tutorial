import torch 
from torch import nn
from torch.utils.data import DataLoader 
from torchvision import datasets 
from torchvision.transforms import ToTensor 


# 1- download dataset 
# 2- create data loader => a class to download in batches
# 3- build model
# 4- train
# 5- save trained model 

BATCH_SIZE=128
EPOCHS=10
LEARNING_RATE=.001

#create the model of our NN
class FeedForwardNet(nn.Module): 
    
    #the constructor 
    def __init__(self): 
        super().__init__()
        #initial layer: Flatten 
        self.flatten=nn.Flatten()
        
        #will have more than one dense layer, Sequeltial allows to pack mutil layers
        self.dense_layers = nn.Sequential(
            #nn.Linear(input_features = 28*28, output_features = 256), #the == a Dense Layer in Keras. input_features = 28*28 because i know the dimensions of the imgs //
            nn.Linear(28*28, 256), 
            nn.ReLu(), #Activation
            #nn.Linear(input_features=256, output_features=10) #why 10 outputs, because we have 10 classes of data
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)
    
    #Allows us to tell Pytorch how to process with the data
    #explain, indicates how to manipulate
    def forward(self, input_data): 
        flatten_data = self.flatten(input_data) #le pasamos el input_data a la capa flatten 
        logits = self.dense_layer(flatten_data)
        predictions = self.softmax(logits)
        
        return predictions

def download_mnist_dataset():
    train_data = datasets.MNIST(
        root ="data", #where to store the dataset, in a new folder wich is going to be created under the directory 
        download = True, #If the dataset of not downloaded, please do it
        train = True, #Telling we only want the train part of 6he dataset
        transform = ToTensor() #let us to aply some tranformation directly to the dataset, 
        #in this case ToTensor => reshape to a new tensor where the valuese are normalize betewn 0 and 1
    )
    
    test_data = datasets.MNIST(
        root ="data", #where to store the dataset, in a new folder wich is going to be created under the directory 
        download = True, #If the dataset of not downloaded, please do it
        train = False, #Telling we only want the test part of 6he dataset
        transform = ToTensor() #let us to aply some tranformation directly to the dataset, 
        #in this case ToTensor => reshape to a new tensor where the valuese are normalize betewn 0 and 1
    )
    
    return train_data, test_data


def train_one_epoch(model, data_loader, loss_fun, optimizer, device): 
    for inputs, targets in data_loader: 
        inputs, targets = inputs.to(device), targets.to(device)
        
        #caculated loss
        predictions =  model(inputs)
        loss = loss_fn(predictions, targets)
        
        #backpropagation loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.setp() #update weights
        
    print(f"Loss: {loss.item()}")
        

def train(model, data_loader, loss_fun, optimizer, device, epochs): 
    for i in range(epochs): 
        print(f"Epoch{i+1}")
        train_one_epoch(model, data_loader, loss_fun, optimizer, device)
        print("---------------------------------------------------------")
    print("Traing is Done!")


if __name__ == "__main__":
    #Step 1
    train_data, _ = download_mnist_dataset()
    
    #Step2: create a data loader fo the train and test set 
    train_data_loader = DataLoader(
        train_data, 
        batch_size = BATCH_SIZE
    )
    
 
    #Step 3: build model
    if torch.cuda.is_available(): 
        device = "cuda"
    else: 
        device = "cpu"
    print(f"Using {device} device")
    
    feed_forward_net = FeedForwardNet().to(device) #to(device) Cuda o CPU where the model will be train. we must cheeck wich aceletarion its avalibel 
    
    #Step 4: traing
    
    #instantiate loss function and optimizer 
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        feed_forward_net.parameters(),
        lr=LEARNING_RATE
        )
    
    train(feed_forward_net, train_data_loader, loss_fun, optimizer, device, EPOCHS)
    
    #Setep 5: Store Model 
    
    torch.save(feed_forward_net.state_dict(), #dictionary of the model that has the important parameters of the model
               "feedforwardnet.pth"
               )
    
    print('Model Trained and Stored at feedforwardnet.pth')
