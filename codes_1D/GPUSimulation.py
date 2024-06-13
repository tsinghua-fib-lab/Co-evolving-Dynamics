import networkx as nx
import numpy as np
import torch
import pickle
from torch import LongTensor, Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected,coalesce
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from scipy.special import softmax
import tqdm
import time
import os
from utils import *


class NodeDynamics(MessagePassing):
    def __init__(self, alpha, sigma, dt, seed, num_A, num_B, update_type, mode, device=torch.device('cpu'), **kwargs):

        super().__init__(aggr='add')


        for k,v in alpha.items():
            if v is not None:
                if k == 'AtoB':
                    self.alpha_AtoB = v
                elif k == 'BtoA':
                    self.alpha_BtoA = v
                elif k == 'A':
                    self.alpha_A = v
                elif k == 'B':
                    self.alpha_B = v
            else:
                pass


        
        self.dt = dt

        self.device = device
        self.seed = seed

        self.num_A = num_A
        self.num_B = num_B

        self.A_offset = num_A 


        if not isinstance(sigma, list):
            self.sigma = sigma
        elif len(sigma) == 2:
            self.sigma = torch.cat([torch.ones(self.num_A).to(self.device)*sigma[0], torch.ones(self.num_B).to(self.device)*sigma[1]]).unsqueeze(-1)
        else:
            pass

        self.update_type = update_type
        self.mode = mode
        self.itc_F = 'linear'


    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        
        bipartite = self.update_type.split('+')[0]
        unipartite = self.update_type.split('+')[1]

        if bipartite == 'Both':
            idx_AB_tmp = ((edge_index[0, :]<self.A_offset)&(edge_index[1, :]>=self.A_offset)).nonzero().squeeze()
            edge_index_AB_tmp = edge_index.index_select(dim = 1, index = idx_AB_tmp)
            
            idx_BA_tmp = ((edge_index[0, :]>=self.A_offset)&(edge_index[1, :]<self.A_offset)).nonzero().squeeze()
            edge_index_BA_tmp = edge_index.index_select(dim = 1, index = idx_BA_tmp)

            edge_indices = torch.cat([edge_index_AB_tmp, edge_index_BA_tmp], dim=1)
            self.alpha = torch.cat([torch.ones_like(edge_index_AB_tmp[0,:])*self.alpha_AtoB, torch.ones_like(edge_index_BA_tmp[0, :])*self.alpha_BtoA])
        elif bipartite == 'AtoB':
            idx_tmp = ((edge_index[0, :]<self.A_offset)&(edge_index[1, :]>=self.A_offset)).nonzero().squeeze()
            edge_index_tmp = edge_index.index_select(dim = 1, index = idx_tmp) 
            edge_indices = edge_index_tmp
            self.alpha = torch.ones_like(edge_index_tmp[0,:])*self.alpha_AtoB
        elif bipartite == 'BtoA':
            idx_tmp = ((edge_index[0, :]>=self.A_offset)&(edge_index[1, :]<self.A_offset)).nonzero().squeeze()
            edge_index_tmp = edge_index.index_select(dim = 1, index = idx_tmp) 
            edge_indices = edge_index_tmp[[1,0], :]
            self.alpha = torch.ones_like(edge_index_tmp[0,:])*self.alpha_BtoA
        else:
            pass

        if 'A' in unipartite:
            idx_tmp = ((edge_index[0, :]<self.A_offset)&(edge_index[1, :]<self.A_offset)).nonzero().squeeze()
            edge_index_tmp = edge_index.index_select(dim = 1, index = idx_tmp) 
            if bipartite != 'X':
                edge_indices = torch.cat([edge_indices, edge_index_tmp], dim=1)
                self.alpha = torch.cat([self.alpha, torch.ones_like(edge_index_tmp[0, :])*self.alpha_A])
            else: 
                edge_indices = edge_index_tmp
                self.alpha = torch.ones_like(edge_index_tmp[0, :])*self.alpha_A
            
        else:
            pass

        if 'B' in unipartite :
            idx_tmp = ((edge_index[0, :]>=self.A_offset)&(edge_index[1, :]>=self.A_offset)).nonzero().squeeze()
            edge_index_tmp = edge_index.index_select(dim = 1, index = idx_tmp) 
            if bipartite != 'X':
                edge_indices = torch.cat([edge_indices, edge_index_tmp], dim=1)
                self.alpha = torch.cat([self.alpha, torch.ones_like(edge_index_tmp[0, :])*self.alpha_B])
            else:
                edge_indices = edge_index_tmp
                self.alpha = torch.ones_like(edge_index_tmp[0, :])*self.alpha_B
            
        else:
            pass
        
        self.alpha = self.alpha.unsqueeze(-1)
        return self.propagate(edge_indices, x=x)
    
    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:

        if self.mode == 'vector':
            
            if self.itc_F == 'linear':
    
                return self.alpha * (x_j - x_i) * self.dt
            else:
                raise NotImplementedError
            
        elif self.mode == 'scalar':

            if self.itc_F == 'linear':
                return self.alpha * (x_j - x_i) * self.dt
            else:
                raise NotImplementedError
        else: 
            raise NotImplementedError

    def update(self, inputs: torch.Tensor, x: torch.Tensor) -> torch.Tensor: 
        if self.mode == 'vector':
            
            if self.itc_F == 'linear':  
                
                tmp = x + inputs + torch.sqrt(self.sigma *(torch.ones_like(x) - torch.square(x)) * self.dt) * torch.normal(mean=0, std=1.0, size=x.shape).to(self.device)
                idc = (torch.abs(tmp) <= 1).float()
                return  (1.0 - idc) * x + idc * tmp
            else:
                raise NotImplementedError

        elif self.mode == 'scalar':
            
            if self.itc_F == 'linear':  
                
                tmp = x + inputs + torch.sqrt(self.sigma *(torch.ones_like(x) - torch.square(x)) * self.dt) * torch.normal(mean=0, std=1.0, size=x.shape).to(self.device)
                idc = (torch.abs(tmp) <= 1).float()
                return  (1.0 - idc) * x + idc * tmp
            
            
            else:
                raise NotImplementedError

class NetworkDynamics:
    def __init__(self, num_A, num_B, gamma_plus, gamma_minus, r_plus, r_minus, dt, sim_F='linear', **kwargs):
        
        self.num_A = num_A
        self.num_B = num_B
        self.A_offset = num_A 

        self.evolving_type = []

        for k,v in gamma_plus.items():
            if v is not None:
                if k == 'AB':
                    self.gamma_plus_AB = v
                    self.gamma_minus_AB = gamma_minus[k]
                    self.r_plus_AB = r_plus[k]
                    self.r_minus_AB = r_minus[k]
                    self.evolving_type.append(k)
                    
                if  k == 'A':
                    self.gamma_plus_A = v
                    self.gamma_minus_A = gamma_minus[k]
                    self.r_plus_A = r_plus[k]
                    self.r_minus_A = r_minus[k]
                    self.evolving_type.append(k)

                if k == 'B':
                    self.gamma_plus_B = v
                    self.gamma_minus_B = gamma_minus[k]
                    self.r_plus_B = r_plus[k]
                    self.r_minus_B = r_minus[k]
                    self.evolving_type.append(k)

        self.dt = dt

        self.N = self.num_A + self.num_B
        self.sim_F = sim_F

    def edge_evolving(self, edge_list, states, init=False):
        if init:
            print(self.sim_F)
            if states.shape[1] == 1: 
                self.similarity = torch.matmul(states, states.T) # (num_A+num_B, num_A+num_B)
            elif states.shape[1] > 1:
                f_n = states.shape[1]
                min_tenor = torch.minimum(torch.matmul(states[:,[0]], states[:,[0]].T), torch.matmul(states[:,[1]], states[:,[1]].T))
                for ff in range(2,f_n):
                    min_tenor = torch.minimum(min_tenor, torch.matmul(states[:,[ff]], states[:,[ff]].T))
                self.similarity = min_tenor


            breaking_edge = self.breaking().t().contiguous()
            connecting_edge = self.connecting().t().contiguous()
            
            breaking_edge_num = breaking_edge.shape[1]
            connecting_edge_num = connecting_edge.shape[1]

            edge_list = torch.cat([connecting_edge, breaking_edge], dim=1)
            attr_idc =  torch.ones_like(edge_list[0, :])
            attr_idc[connecting_edge_num:] = -1
            edge_list, attr_idc  = coalesce(edge_index = edge_list, edge_attr = attr_idc, num_nodes=self.num_A+self.num_B, reduce="add") 
            edge_list = edge_list[:, attr_idc > 0] 
            return edge_list

        else:
            if  states.shape[1] == 1:
                self.similarity = torch.matmul(states, states.T) # (num_A+num_B, num_A+num_B)
            elif states.shape[1] > 1:
                f_n = states.shape[1]
                min_tenor = torch.minimum(torch.matmul(states[:,[0]], states[:,[0]].T), torch.matmul(states[:,[1]], states[:,[1]].T))
                for ff in range(2,f_n):
                    min_tenor = torch.minimum(min_tenor, torch.matmul(states[:,[ff]], states[:,[ff]].T))
                self.similarity = min_tenor
                

            breaking_edge = self.breaking().t().contiguous()
            connecting_edge = self.connecting().t().contiguous()




            breaking_edge_num = breaking_edge.shape[1]
            connecting_edge_num = connecting_edge.shape[1]
            edge_list_num = edge_list.shape[1]



            # existing edges
            existing_edges = torch.cat([edge_list, breaking_edge], dim=1)
            existing_idc = torch.ones_like(existing_edges[0, :])
            existing_idc[edge_list_num:] = -1.0
            existing_edges, existing_idc  = coalesce(edge_index = existing_edges, edge_attr = existing_idc, num_nodes=self.num_A+self.num_B, reduce="add") 
            existing_edges = existing_edges[:, existing_idc>0.0]

            adding_edges = torch.cat([edge_list, connecting_edge], dim=1)
            adding_idc = torch.ones_like(adding_edges[0, :])
            adding_idc[:edge_list_num] = -1.0
            adding_edges, adding_idc = coalesce(edge_index = adding_edges, edge_attr = adding_idc, num_nodes=self.num_A+self.num_B, reduce="add") 
            adding_edges = adding_edges[:, adding_idc>0.0]

            edge_list = torch.cat([existing_edges, adding_edges], dim=1)

            return edge_list

  
    def breaking(self): # edge_list: (population, location)
        # A*A, B*B, A*B
        breaking_edge = {}
        if self.sim_F == 'linear':
            if 'A' in self.evolving_type:
                breaking_p_A = self.gamma_minus_A * (1.0 + self.r_minus_A * self.similarity[:self.A_offset, :self.A_offset]) * self.dt
                breaking_edge['A'] = (torch.rand_like(breaking_p_A) <= breaking_p_A).nonzero()
                breaking_edge['A'] = breaking_edge['A'][breaking_edge['A'][:,0] != breaking_edge['A'][:,1], :]
                
            if 'B' in self.evolving_type:
                breaking_p_B = self.gamma_minus_B * (1.0 + self.r_minus_B * self.similarity[self.A_offset:, self.A_offset:]) * self.dt
                breaking_edge['B'] = (torch.rand_like(breaking_p_B) <= breaking_p_B).nonzero() 
                breaking_edge['B'] = breaking_edge['B'] + self.A_offset
                breaking_edge['B'] = breaking_edge['B'][breaking_edge['B'][:,0] != breaking_edge['B'][:,1], :]
                
            if 'AB' in self.evolving_type:
                breaking_p_AB = self.gamma_minus_AB * (1.0 + self.r_minus_AB * self.similarity[:self.A_offset, self.A_offset:]) * self.dt
                breaking_edge_AB = (torch.rand_like(breaking_p_AB) <= breaking_p_AB).nonzero()
                breaking_edge_AB[:, 1] = breaking_edge_AB[:, 1] + self.A_offset

                breaking_p_BA = self.gamma_minus_AB * (1.0 + self.r_minus_AB * self.similarity[self.A_offset:, :self.A_offset]) * self.dt
                breaking_edge_BA = (torch.rand_like(breaking_p_BA) <= breaking_p_BA).nonzero()
                breaking_edge_BA[:, 0] = breaking_edge_BA[:, 0] + self.A_offset

                breaking_edge['AB'] = torch.cat([breaking_edge_AB, breaking_edge_BA], dim=0)
        else:
            raise NotImplementedError



        return torch.cat([v for v in breaking_edge.values()], dim=0).long()

    def connecting(self):
        connecting_edge = {}
        if self.sim_F == 'linear':
            if 'A' in self.evolving_type:
                connecting_p_A = self.gamma_plus_A * (1.0 + self.r_plus_A * self.similarity[:self.A_offset, :self.A_offset])/(self.num_A) * self.dt
                connecting_edge['A'] = (torch.rand_like(connecting_p_A) <= connecting_p_A).nonzero()
                connecting_edge['A'] = connecting_edge['A'][connecting_edge['A'][:,0] != connecting_edge['A'][:,1], :]
               
            if 'B' in self.evolving_type:
                connecting_p_B = self.gamma_plus_B * (1.0 + self.r_plus_B * self.similarity[self.A_offset:, self.A_offset:])/(self.num_B) * self.dt
                connecting_edge['B'] = (torch.rand_like(connecting_p_B) <= connecting_p_B).nonzero()
                connecting_edge['B'] = connecting_edge['B'] + self.A_offset
                connecting_edge['B'] = connecting_edge['B'][connecting_edge['B'][:,0] != connecting_edge['B'][:,1], :]
                
            if 'AB' in self.evolving_type:
                connecting_p_AB = self.gamma_plus_AB * (1.0 + self.r_plus_AB * self.similarity[:self.A_offset, self.A_offset:])/np.sqrt(self.num_A*self.num_B)/2 * self.dt
                connecting_edge_AB = (torch.rand_like(connecting_p_AB) <= connecting_p_AB).nonzero()
                connecting_edge_AB[:, 1] = connecting_edge_AB[:, 1] + self.A_offset

                connecting_p_BA = self.gamma_plus_AB * (1.0 + self.r_plus_AB * self.similarity[self.A_offset:, :self.A_offset])/np.sqrt(self.num_A*self.num_B)/2 * self.dt
                connecting_edge_BA = (torch.rand_like(connecting_p_BA) <= connecting_p_BA).nonzero()
                connecting_edge_BA[:, 0] = connecting_edge_BA[:, 0] + self.A_offset

                connecting_edge['AB'] = torch.cat([connecting_edge_AB, connecting_edge_BA], dim=0)
        else:
            raise NotImplementedError

        return torch.cat([v for v in connecting_edge.values()], dim=0).long()




def bipartite_simulation(
    node_path: str, 
    edge_path: str,
    update_type: str,
    mode: str,
    num_feature: int, 
    alpha: dict,  
    gamma_plus: dict,
    gamma_minus: dict,
    sigma: list,
    r_plus: dict,
    r_minus:dict,
    dt: float,
    overall_time: float,
    seed: int, 
    device: str,
    network_sim_F='linear',
    extended=None):
    

    bidataset = BipartiteDataset(node_path=node_path, edge_path=edge_path, num_feature=num_feature, update_type=update_type)


    num_A = bidataset.num_A
    num_B = bidataset.num_B
    num_runs = int(overall_time//dt)+1

    gp_keywords = [k+'{:.2f}'.format(v*1000) for k,v in gamma_plus.items() if v is not None]
    gm_keywords = [k+'{:.2f}'.format(v*1000) for k,v in gamma_minus.items() if v is not None]
    a_keywords = [k+'{:.2f}'.format(v*1000) for k,v in alpha.items() if v is not None]
    rp_keywords = [k+'{:.2f}'.format(v) for k,v in r_plus.items() if v is not None]
    rm_keywords = [k+'{:.2f}'.format(v) for k,v in r_minus.items() if v is not None]

    gp_keywords = '-'.join(gp_keywords)
    gm_keywords = '-'.join(gm_keywords)
    a_keywords = '-'.join(a_keywords)
    rp_keywords = '-'.join(rp_keywords)
    rm_keywords = '-'.join(rm_keywords)

    init_method = node_path.split('_')[1][:-4]

    if init_method != 'random':
        print(init_method)

        if not isinstance(sigma, list):
            output_path = './{}_results_rp{}_rm{}/alpha:{},gp:{},gm:{},s:{:.2f}/'.format(init_method, rp_keywords, rm_keywords, a_keywords, gp_keywords, gm_keywords, sigma*1000)
            print(output_path)
        elif len(sigma) == 2:
            output_path = './{}_results_rp{}_rm{}/alpha:{},gp:{},gm:{},s:{:.2f}/'.format(init_method, rp_keywords, rm_keywords, a_keywords, gp_keywords, gm_keywords, max(sigma)*1000)
            print(output_path)


        if not os.path.exists('./{}_results_rp{}_rm{}/'.format(init_method, rp_keywords, rm_keywords)):
            os.mkdir('./{}_results_rp{}_rm{}/'.format(init_method, rp_keywords, rm_keywords))

        if not os.path.exists(output_path):
            os.mkdir(output_path)
    
    else: 

        if not isinstance(sigma, list):
            output_path = './results_rp{}_rm{}/alpha:{},gp:{},gm:{},s:{:.2f}/'.format(rp_keywords, rm_keywords, a_keywords, gp_keywords, gm_keywords, sigma*1000)
            print(output_path)
        elif len(sigma) == 2:
            output_path = './results_rp{}_rm{}/alpha:{},gp:{},gm:{},s:{:.2f}/'.format(rp_keywords, rm_keywords, a_keywords, gp_keywords, gm_keywords, max(sigma)*1000)
            print(output_path)


        if not os.path.exists('./results_rp{}_rm{}/'.format(rp_keywords, rm_keywords)):
            os.mkdir('./results_rp{}_rm{}/'.format(rp_keywords, rm_keywords))

        if not os.path.exists(output_path):
            os.mkdir(output_path)


    CD = NodeDynamics(alpha=alpha, sigma=sigma, dt=dt, seed=seed,  num_A=num_A, num_B=num_B, device=device, update_type=update_type, mode=mode)
    ND = NetworkDynamics(num_A = num_A, num_B = num_B,  gamma_plus = gamma_plus, gamma_minus = gamma_minus, r_plus = r_plus, r_minus=r_minus, dt=dt, sim_F=network_sim_F)
    
    
    states = torch.FloatTensor(bidataset.node_feature).to(device)
    edge_list = ND.edge_evolving(edge_list=bidataset.edge_index, states=states, init=True)

    local_states = states.cpu().numpy()
    local_edges = edge_list.cpu().numpy().T

    np.save(output_path+'edge_run{:d}.npy'.format(0), local_edges)
    np.save(output_path+'node_run{:d}.npy'.format(0), local_states)

    avg_list = [(0, np.mean(local_states))]
    for t in tqdm.tqdm(range(num_runs)):
        
        states = CD(states, edge_list)
        edge_list = ND.edge_evolving(edge_list=edge_list, states=states)
        


        if (t+1)%1000==0:
            local_states = states.cpu().numpy()
            local_edges = edge_list.cpu().numpy().T
            avg_list.append((t+1, np.mean(local_states)))
            np.save(output_path+'edge_run{:d}.npy'.format(t+1), local_edges)
            np.save(output_path+'node_run{:d}.npy'.format(t+1), local_states)
            
    avg_list.append((t+1, np.mean(local_states)))
    np.save(output_path+'edge_run{:d}.npy'.format(t+1), local_edges)
    np.save(output_path+'node_run{:d}.npy'.format(t+1), local_states)

    np.save(output_path+'recorded.npy', np.array(avg_list))
    return 
