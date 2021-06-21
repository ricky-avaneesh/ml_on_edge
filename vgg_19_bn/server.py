#IMPORTING HELPER MODULES
from flask import Flask, json , request , jsonify
import flask
import numpy as np
import requests
import time
#IMPORT PYTORCH AND RELATED MODULES
import torch
import torch.nn as nn
import torchvision.models as models

#------------------------------------------------------------------------------------------------------------
#INITIALISING FLASK OBJECT
app = Flask(__name__)
print("[SERVER] has been started")

#------------------------------------------------------------------------------------------------------------
#HELPER FUNCTIONS
#GET ALL LAYERS OF THE MODEL
def get_children(model):
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



#------------------------------------------------------------------------------------------------------------
#DEFINING TAIL NETWORK
class TailNet(nn.Module):
    def __init__(self ,layers_of_network , random_layer ):
        super( TailNet, self ).__init__()
        print("---------------------------------------------------------")
        print("[SPLITTING] the model for tail part...")
        print("---------------------------------------------------------")
        self.layers_useful = layers_of_network[random_layer+1:]

    def forward(self , input_tensor):
        for i in self.layers_useful:
            try:
                input_tensor = i(input_tensor)
            except:
                input_tensor = input_tensor.view(1,-1)
                input_tensor = i(input_tensor)
        return input_tensor

#------------------------------------------------------------------------------------------------------------
#LOADING MODEL IN MEMORY
model = models.vgg19_bn(pretrained=True)
model.eval()
layers_of_network = get_children(model)
#------------------------------------------------------------------------------------------------------------
#DEFIFING ROUTES
@app.route("/tail_prediction" , methods=['POST'])
def tail_prediction():
    print("\n")
    print("---------------------------------------------------------")
    print("[REQUEST]  received from client for single image")
    print("---------------------------------------------------------")
    data = request.json
    time_upload = time.time() - data["time_client_sends_activation"]
    time_server_inference_start = time.time()
    i = data["i"]
    activations = np.array(data["activation"])
    activations = torch.from_numpy(activations)
    print("[SHAPE] of activations received from server")
    print("---------------------------------------------------------")
    print(activations.shape)
    print("---------------------------------------------------------")
    print("[PREDICTING] output from intermediate activations")
    tailnetwork = TailNet(layers_of_network , i)
    with torch.no_grad():
        output = tailnetwork(activations.float())
    print("[OUTPUT] of the network")
    print("---------------------------------------------------------")
    print(output.shape)
    print("---------------------------------------------------------")
    print("[NORMALISING] the outputs")
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print("---------------------------------------------------------")
    print("[PREDICTING] the labels")
    print("---------------------------------------------------------")
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
    print("---------------------------------------------------------")
    print("[SENDING] the prediction to client")
    print("---------------------------------------------------------")
    time_inference_server_stop = time.time() - time_server_inference_start
    data_ = {'category' : categories[top5_catid[0]] , 'confidence_score' : top5_prob[0].item() , 'time_upload' : time_upload , 'time_inference_server': time_inference_server_stop , 'time_server_sends_client':time.time()}
    return json.dumps(data_)   

    
#------------------------------------------------------------------------------------------------------------
#STARTING THE SERVER
if __name__=='__main__':
    app.run()


