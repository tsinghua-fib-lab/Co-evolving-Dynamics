import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset


def get_angular_similarity(x,y):
    cs = np.sum(x*y, axis=1)/np.linalg.norm(x, axis=1)/np.linalg.norm(y, axis=1)
    cs = np.clip(cs, -1, 1)
    return (1.0 - 2*np.arccos(cs)/np.pi).reshape(-1,1)

def get_average_sim(edge,states):
    sim = []
    for e in edge:
        sim.append(np.sum(states[e[0],:]*states[e[1],:]))
    return np.mean(sim)

class BipartiteDataset(Dataset):
    def __init__(self, node_path, edge_path, num_feature, update_type):
        super(BipartiteDataset, self).__init__()
        self.node_path = node_path
        self.edge_path = edge_path
        self.num_feature = num_feature
        self.update_type = update_type
        self._load_data()
    
    def _load_data(self):
        print("--- load data start ---")
        # load node feature, csv, columns, name (A0, A1; B0, B1), type (A, B), f0, f1, f2, f3,
        print(self.node_path)
        node = pd.read_csv(self.node_path)[['name', 'type'] + ['f{:d}'.format(s) for s in range(self.num_feature)]]
        self.num_A = node.loc[node['type'] == 'A']['name'].drop_duplicates().shape[0]
        self.num_B = node.loc[node['type'] == 'B']['name'].drop_duplicates().shape[0]
        print("num_nodes: Set A {:d}, Set B {:d}".format(self.num_A, self.num_B))
        
        node['name'] = node['name'].transform(lambda x: int(x[1:]) if x[0]== 'A' else int(x[1:]) + self.num_A)
        node = node.sort_values(by='name')
        print(node['name'].max())
        self.node_feature = node[['f{:d}'.format(s) for s in range(self.num_feature)]].to_numpy()
        print("feature:{:d}".format(self.node_feature.shape[1]))
        # load edge feature, columns: A, B, update_type: AtoB or BtoA or Both
        bipartite = self.update_type.split('+')[0]
        unipartite = self.update_type.split('+')[1]
        #edge = pd.read_csv(self.edge_path)[['source', 'target', 'type']]
        print("Bipartite Type: {}".format(bipartite))
        print("Unipartite Type: {}".format(unipartite))

        # edge_list = []
        # if bipartite == 'Both':
        #     edge_tmp = edge.loc[edge['type']=='Both'][['source', 'target']]
        #     edge_tmp['source'] = edge_tmp['source'].transform(lambda x: int(x[1:]))
        #     edge_tmp['target'] = edge_tmp['target'].transform(lambda x: int(x[1:])+self.num_A)
        #     edge_tmp = edge_tmp.to_numpy()
        #     edge_list.append(edge_tmp)
        #     print("Bipartite-{}-{}".format(bipartite, edge_tmp.shape[0]))
        # elif bipartite == 'AtoB':
        #     edge_tmp = edge.loc[edge['type']=='AtoB'][['source', 'target']]
        #     edge_tmp['source'] = edge_tmp['source'].transform(lambda x: int(x[1:]))
        #     edge_tmp['target'] = edge_tmp['target'].transform(lambda x: int(x[1:])+self.num_A)
        #     edge_tmp = edge_tmp.to_numpy()
        #     edge_list.append(edge_tmp)
        #     print("Bipartite-{}-{}".format(bipartite, edge_tmp.shape[0]))
        # elif bipartite == 'BtoA':
        #     edge_tmp = edge.loc[edge['type']=='BtoA'][['source', 'target']]
        #     edge_tmp['source'] = edge_tmp['source'].transform(lambda x: int(x[1:]))
        #     edge_tmp['target'] = edge_tmp['target'].transform(lambda x: int(x[1:])+self.num_A)
        #     edge_tmp = edge_tmp.to_numpy()
        #     edge_list.append(edge_tmp)
        #     print("Bipartite-{}-{}".format(bipartite, edge_tmp.shape[0]))
        # else:
        #     pass

        # if 'A' in unipartite:
        #     edge_tmp = edge.loc[edge['type']=='A'][['source', 'target']]
        #     edge_tmp['source'] = edge_tmp['source'].transform(lambda x: int(x[1:]))
        #     edge_tmp['target'] = edge_tmp['target'].transform(lambda x: int(x[1:]))
        #     edge_tmp = edge_tmp.to_numpy()
        #     edge_list.append(edge_tmp)
        #     print("Uipartite-{}-{}".format('A', edge_tmp.shape[0]))
        # else:
        #     pass

        # if 'B' in unipartite :
        #     edge_tmp = edge.loc[edge['type']=='B'][['source', 'target']]
        #     edge_tmp['source'] = edge_tmp['source'].transform(lambda x: int(x[1:])+self.num_A)
        #     edge_tmp['target'] = edge_tmp['target'].transform(lambda x: int(x[1:])+self.num_A)
        #     edge_tmp = edge_tmp.to_numpy()
        #     edge_list.append(edge_tmp)
        #     print("Uipartite-{}-{}".format('B', edge_tmp.shape[0]))
        # else:
        #     pass
        
        # print(np.min(np.concatenate(edge_list, axis=0), axis=0))
        # self.edge_index = np.concatenate(edge_list, axis=0)
        self.edge_index = []

            
    def __len__(self):
        return self.num_A + self.num_B
    
    def __getitem__(self, index):
        return self.node_feature, self.edge_index

        


