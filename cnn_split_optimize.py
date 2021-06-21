import json
from math import ceil
import numpy as np

def load_settings():
    with open('settings.json', 'r') as settings_file:
        settings_data = settings_file.read()

    # parse file
    settings = json.loads(settings_data)
    return settings

def memory_requirement(nlayers):

    layers = ceil(nlayers)
    kernel_prod = (settings['kernel_w'] * settings['kernel_h'])
    weights = layers * kernel_prod

    for i in range(1, layers):
        depth = layers - i
        weights += depth * kernel_prod * (depth + 1)

    return (weights * 32)*1.25e-7

def getallindices(search_list, x):
    index_list = []

    for i in range(len(search_list)):
        if search_list[i] == x:
            index_list.append(i)
    
    return index_list

settings = load_settings()

objs = [
    lambda x: memory_requirement(x[0])/(settings['edge_cores']*settings['cpu_size']) + memory_requirement(1)/(settings['bandwidth']) + memory_requirement(x[1])/(settings['server_cores']*settings['server_cpu']),
    lambda x: -1 * memory_requirement(x[0])
]

server_layer_list = []
edge_layer_list = []
latency_list = []
memory_list = []

for i in range(1, settings['layers']):
    for j in range(1, settings['layers']):
        if i + j != settings['layers'] or memory_requirement(i) >= settings['memory_size']*.1:
            continue

        latency = objs[0]([i, j])
        memory_usage = objs[1]([i, j])

        # print(i, j, latency, memory_usage)

        latency_list.append(latency)
        memory_list.append(memory_usage)
        edge_layer_list.append(i)
        server_layer_list.append(j)

min_latency = min(latency_list)
latency_indices = getallindices(latency_list, min_latency)
memory_min = np.inf
min_index = latency_indices[0]

for i in latency_indices:
    if memory_list[i] < memory_min:
        memory_min = memory_list[i]
        min_index = i

print(latency_list[min_index], memory_list[min_index], edge_layer_list[min_index], server_layer_list[min_index])