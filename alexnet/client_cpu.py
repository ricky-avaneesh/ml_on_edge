#IMPORTING REQUIRED HELPER MODULES
import random
import time
import requests
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#Setting figure size
plt.rcParams["figure.figsize"] = (20,3)

#IMPORTING PYTORCH MODULES
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset , DataLoader


#------------------------------------------------------------------------------------------------------------
#HELPER FUNCTIONS 
def load_settings():
    with open('settings.json', 'r') as settings_file:
        settings_data = settings_file.read()

    # parse file
    settings = json.loads(settings_data)
    return settings
def get_children(model):
    """
    INPUT: Model
    OUTPUT : All layers present in the model
    """
    children = list(model.children())
    flatt_children = []
    if children == []:
        return model
    else:
        for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children

def generate_random_number(num_layers):
    """
    INPUT : Layers of Model
    OUTPUT : Random Layer Number from 0 to len(num_layers) - 2 
    """
    return random.randint(0, num_layers-2)

#------------------------------------------------------------------------------------------------------------
#LOADING THE MODEL IN MEMORY
model = models.alexnet(pretrained=True)

device = torch.device("cpu")

model.to(device)
model.eval()
layers_of_network = get_children(model)
#------------------------------------------------------------------------------------------------------------
#HEAD NETWORK CLASS
class HeadNet(nn.Module):
    def __init__(self ,layers_of_network , random_layer ):
        super( HeadNet, self ).__init__()
        print("---------------------------------------------------------")
        print("[SPLITTING] the model for head part")
        print("---------------------------------------------------------")
        self.layers_useful = layers_of_network[0:random_layer+1]

    def forward(self , input_tensor):
        print("[COMPUTING]the activation from head part")
        print("---------------------------------------------------------")
        for i in self.layers_useful:
            try:
                input_tensor = i(input_tensor)
            except:
                input_tensor = input_tensor.view(1,-1)
                input_tensor = i(input_tensor)
        return input_tensor

#------------------------------------------------------------------------------------------------------------
# MAIN FUNCTIONS

def single_image_prediction(image_path):
    print("---------------------------------------------------------")
    print(f"[READING] the image {image_path}")
    #PROCESSING THE INPUT IMAGE
    input_image = Image.open(image_path)
    print("---------------------------------------------------------")
    print("[PREPROCESSING] input image")
    preprocess = transforms.Compose([
        transforms.Resize(256) , transforms.CenterCrop(224) ,transforms.ToTensor() , transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    print("---------------------------------------------------------")
    print("[IMAGE] preprocessed")
    print("---------------------------------------------------------")
    print(f"[TOTAL] layers of network - {len(layers_of_network)}")
    time_inference_client_start = time.time()
    print("---------------------------------------------------------")
    print("[GENERATING] a random number")
    print("---------------------------------------------------------")
    random_layer = generate_random_number(len(layers_of_network))
    print(f"[GENERATED] random number - {random_layer}")
    headnetwork = HeadNet(layers_of_network , random_layer)
    print("[HEAD PREDICTION] ready to be performed")
    print("----------------------------------------------------------")
    with torch.no_grad():      
        activation = headnetwork(input_batch)
    print("[SHAPE] of intermediate activations from head")
    print("---------------------------------------------------------")
    print(activation.shape)
    activation = activation.numpy().tolist()
    print("---------------------------------------------------------")
    print("[SENDING] the activations at split to the server")
    print("---------------------------------------------------------")
    time_inference_total_client = time.time() - time_inference_client_start
    data = {'i': random_layer , 'activation' : activation , 'time_client_sends_activation' : time.time()}
    response = requests.post(url+"/tail_prediction", json = data)
    print("[RECEIVED] results from the server")
    print("---------------------------------------------------------")
    result = json.loads(response.text)
    download_time = time.time() - result["time_server_sends_client"]
    print(f"Prediction is {result['category']} with confidence {result['confidence_score']} and inference time on edge {time_inference_total_client} , upload time {result['time_upload']} , inference time on server {result['time_inference_server']} , download time {download_time}")
    print("---------------------------------------------------------")


def time_image_predictions(image_path):
    print("---------------------------------------------------------")
    print(f"[READING] the image {image_path}")
    #PROCESSING THE INPUT IMAGE
    input_image = Image.open(image_path)
    print("---------------------------------------------------------")
    print("[PREPROCESSING] input image")
    preprocess = transforms.Compose([
        transforms.Resize(256) , transforms.CenterCrop(224) ,transforms.ToTensor() , transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    print("---------------------------------------------------------")
    print("[IMAGE] preprocessed")
    print("---------------------------------------------------------")
    print(f"[TOTAL] layers of network - {len(layers_of_network)}")
    total_time_list = []
    for i in range(len(layers_of_network)-1):
        time_inference_client_start = time.time()
        headnetwork = HeadNet(layers_of_network , i)
        print("[HEAD PREDICTION] ready to be performed")
        print("----------------------------------------------------------")
        with torch.no_grad():      
            activation = headnetwork(input_batch)
        print("[SHAPE] of intermediate activations from head")
        print("---------------------------------------------------------")
        print(activation.shape)
        activation = activation.numpy().tolist()
        print("---------------------------------------------------------")
        print("[SENDING] the activations at split to the server")
        print("---------------------------------------------------------")
        time_inference_total_client = time.time() - time_inference_client_start
        data = {'i': i , 'activation' : activation , 'time_client_sends_activation' : time.time()}
        response = requests.post(url+"/tail_prediction", json = data)
        print("[RECEIVED] results from the server")
        print("---------------------------------------------------------")
        result = json.loads(response.text)
        download_time = time.time() - result["time_server_sends_client"]
        total_time = download_time + time_inference_total_client + result['time_upload'] + result['time_inference_server']
        print(f"[TOTAL TIME is {total_time}")
        total_time_list.append((i , total_time))
        print("---------------------------------------------------------")
    print(total_time_list)
    plt.scatter(*zip(*total_time_list))
    plt.xticks(np.arange(0,len(layers_of_network) - 1, 1.0))
    plt.title("Time for CNN splits")
    plt.xlabel("SPLIT LAYER NUMBER")
    plt.ylabel("TOTAL TIME")
    plt.show()

#------------------------------------------------------------------------------------------------------------
#URL FOR SERVER
url = load_settings()["server_url"]
#------------------------------------------------------------------------------------------------------------
#WHILE LOOP
request_again= True
while request_again:
    toDo = int(input("[ENTER] choice- \n\n 1) Random Splitting on single image \n 2) Time on a image(For all splits) \n"))
    if toDo == 1:
        print("---------------------------------------------------------")
        image_path = input("[ENTER] image path \n")
        single_image_prediction(image_path)
    if toDo == 2:
        print("---------------------------------------------------------")
        image_path = input("[ENTER] image path \n")
        time_image_predictions(image_path)



