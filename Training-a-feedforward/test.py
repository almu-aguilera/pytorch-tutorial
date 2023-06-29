import torch 
from train import FeedForwardNet, download_mnist_dataset


class_mapping=[
    "0", 
    "1", 
    "2", 
    "3", 
    "4", 
    "5", 
    "6", 
    "7", 
    "8", 
    "9"
]


def predict(model, data, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(data)
        #tensor objets with a specifics=> Tensor(1, 10)-> [[0.1,0.1,...,0.6]] theindex, will correspond with the class 
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping(predicted_index)
        expected = class_mapping(target)
    
    return predicted, expected

if __name__ == "__main__":    
    #load back the model
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth")

    feed_forward_net.load_state_dict(state_dict) #loaded back the model 
    
    #load MNIST validation Dataset
    _, validation_data = download_mnist_dataset()
    
    #get a sample from the validation dataset for inference 
    input_data, target = validation_data[0][0], validation_data[0][1]
    
    #make a inference 
    predicted, expected = predict(feed_forward_net, input_data, target, class_mapping)
    
    print(f"Predicted: '{predicted}, expected: '{expected}'")
    
    