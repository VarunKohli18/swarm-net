import os
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
import torch_geometric.nn as gnn

from time import time
from tqdm import tqdm
from copy import deepcopy
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


def patch(datapoint: np.ndarray, patch_size: int):
    """
    add a random patch of noise to the input data contiguously from 0 to `patch_size` 
    """
    datapoint = np.array(datapoint.copy())
    loc = 0
    datapoint[loc:loc+patch_size] = np.array([np.random.randint(1,256) for i in range(patch_size)])/255
    return datapoint

def get_pyg_data(scenario: str, processed_data: dict, mask: int = -1):
    """
    load data from the processed data dictionary and return a list of pytorch-geometric Data objects
    """
    x = torch.from_numpy(np.stack([processed_data[scenario][node] for node in nodes])).transpose(0,1).float()
    y = torch.zeros((NUM_NODES)).long()

    if "AN" in scenario:
        anomalies = [int(i) for i in scenario[2:]]
        y[anomalies] = 1

    data_ = []
    for i in range(len(x)): 
        if mask != -1:
            # print(x[i].shape, x[i].numpy()); raise
            x[i] = torch.from_numpy(np.stack([patch(x[i][j].numpy(), mask) for j in range(NUM_NODES)]))
        data_.append(Data(x=x[i], y=y, edge_index=EDGE_INDEX))

    return data_

def get_fullbatch(scenario: str, processed_data: dict):
    """
    returns a full batch of data for a given scenario as a pytorch-geometric DataBatch object 
    """
    data_ = get_pyg_data(scenario, processed_data)
    return Batch.from_data_list(data_)


class GraphModel(torch.nn.Module):
    def __init__(self, layer_type, input_dim, latent_dim, dropout=0.5):
        super().__init__()
        layer = getattr(gnn, layer_type)

        self.conv1 = layer(input_dim, 2*latent_dim)
        self.conv2 = layer(2*latent_dim, latent_dim)

        self.lin = nn.Linear(latent_dim, input_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)

        return x

def get_indices(data, num_nodes):
    """
    returns a list of indices for each node type (0-3 or 0-5 for swarm-1 and swarm-2)
    used for computing thresholds and detection rates for each node
    """
    ld = len(data)
    return [range(i, ld, num_nodes) for i in range(num_nodes)]

def get_thresholds(model, data, num_nodes):
    """
    computes the detection thresholds for each node in using the given data, 
    using cosine similarity as the threshold metric
    """
    data = data.to(next(model.parameters()).device)
    out = model(data.x, data.edge_index)
    indices = get_indices(out, num_nodes)
    thresholds = [0.999 * F.cosine_similarity(out[indices[i]], data.x[indices[i]]).min() for i in range(num_nodes)]
    return thresholds



if __name__ == "__main__":

    # example script usage: python train.py --dataset_name swarm-1 --device 0 --epochs 200 --latent 64 --dropout 0.5 --lr 5e-3 --noise 0.4 --runs 20 --layer_type TransformerConv


    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset file", choices=["swarm-1", "swarm-2"])
    parser.add_argument("--device", type=int, default=-1, help="GPU id for training, pass -1 for cpu")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--layer_type", type=str, default="TransformerConv", help="GNN inner layer, should be a valid torch_geometric.nn layer (Validated layers: GCNConv, GATConv, TransformerConv)")
    parser.add_argument("--latent", type=int, default=64, help="GNN inner layer latent dim")
    parser.add_argument("--dropout", type=float, default=0.1, help="GNN dropout")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--noise", type=float, default=0.4, help="Noise coefficient")
    parser.add_argument("--fake_mask_size", type=int, default=-1, help="Size of random mask for generating fake data during training, set to -1 to disable")
    parser.add_argument("--runs", type=int, help="Number of runs to average performance over", default=1)
    parser.add_argument("--no_save", action="store_true", help="Do not save results table")
    args = parser.parse_args()

    try:
        layer = getattr(gnn, args.layer_type)
    except AttributeError as e:
        print(f"Invalid layer type: {args.layer_type}, Please pass a valid torch_geometric.nn layer (Validated layers: GCNConv, GATConv, TransformerConv)")
        exit(0)

    print("Running with args:", args)

    with open(f"../data/{args.dataset_name}.pkl", "rb") as inp:
        data = pickle.load(inp)

    scenarios = list(data.keys())
    nodes = list(data['D1'].keys())
    NUM_NODES = len(nodes)

    thresholds = []

    for node in nodes:
        t = data['D1'][node]['threshold']
        if t%2 != 0:
            thresholds.append(t-1)
        else:
            thresholds.append(t)

    max_threshold = 2048

    processed = {scenario:{} for scenario in scenarios}
    for scenario in scenarios:
        for i in range(NUM_NODES):
            processed[scenario][nodes[i]] = data[scenario][nodes[i]]['SRAM'][:,:thresholds[i]]/255
            padding = np.zeros((processed[scenario][nodes[i]].shape[0], max_threshold-processed[scenario][nodes[i]].shape[1]))
            processed[scenario][nodes[i]] = np.concatenate((processed[scenario][nodes[i]], padding), axis=1)
    

    # setup the graph structure (adjacency matrix and edge index) 
    A = torch.zeros((NUM_NODES,NUM_NODES))
    
    for i in range(NUM_NODES): A[0,i] = 1

    for i in range(1,NUM_NODES-1): A[i,i+1] = 1
    
    if args.dataset_name == "swarm2": A[3,4] = 0
    
    EDGE_INDEX = torch.stack(torch.where(A==1))

    # setup dataloaders for training
    training_scenarios = ["D1", "D2"]

    train_data_clean = [get_pyg_data(scenario, processed) for scenario in training_scenarios]
    train_data_clean = [x for xs in train_data_clean for x in xs]

    train_data_fake = [get_pyg_data(scenario, processed, mask=args.fake_mask_size) for scenario in training_scenarios]
    train_data_fake = [x for xs in train_data_fake for x in xs[:int(len(xs))]]

    ctrainloader = DataLoader(train_data_clean, batch_size=len(train_data_clean), shuffle=True)
    ftrainloader = DataLoader(train_data_fake, batch_size=len(train_data_fake), shuffle=True)

    # set cuda device if available
    DEVICE = f"cuda:{args.device}" if torch.cuda.is_available() and args.device != -1 else "cpu"
    
    # main training runs (results will be averaged over runs)
    MAIN_ADR = []
    for run in range(1,args.runs+1):
        model = GraphModel(layer_type=args.layer_type, input_dim=max_threshold, latent_dim=args.latent, dropout=args.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        delta = 0.05
        margin = args.fake_mask_size != -1

        ftrain_fullbatch = next(iter(ftrainloader)).to(DEVICE) 
        ctrain_fullbatch = next(iter(ctrainloader)).to(DEVICE)

        x = torch.randn(NUM_NODES, max_threshold)
        edge_index = torch.randint(NUM_NODES, size=EDGE_INDEX.size())

        if run == 1: 
            print(gnn.summary(model, x, edge_index))
            print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        model = model.to(DEVICE)

        model.train()
        minloss = 1e9
        epoch_times = []
        pbar = tqdm(range(1,args.epochs+1), desc=f'Training run {run:2d}')
        for epoch in pbar:
            start = time()
            losses = []
            for data in ctrainloader:
                data = data.to(DEVICE)
                orig_x = data.x.clone()
                data.x = data.x + args.noise*torch.randn_like(data.x)

                out = model(data.x, data.edge_index)
                loss = F.mse_loss(out, orig_x)

                if margin:
                    # compute hinge margin loss: relu(delta + cosine_similarity - threshold)
                    thresholds = get_thresholds(model, ctrain_fullbatch, NUM_NODES)
                    indices = get_indices(ftrain_fullbatch.x, NUM_NODES)

                    data = ftrain_fullbatch.to(DEVICE)
                    out = model(data.x, data.edge_index)
                    cs = F.cosine_similarity(out, ftrain_fullbatch.x)

                    for i in range(NUM_NODES):
                        loss += F.relu(cs[indices[i]] - thresholds[i] + delta).mean()

                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss < minloss:
                    # saving best weights
                    best_weights = deepcopy(model.state_dict())

            stop = time()
            epoch_times.append(stop-start)

            scheduler.step()

            if epoch%5 == 0 or epoch == 1:
                if margin:
                    thresholds = get_thresholds(model, ctrain_fullbatch, NUM_NODES)
                    indices = get_indices(ftrain_fullbatch.x, NUM_NODES)

                    data = ftrain_fullbatch.to(DEVICE)
                    out = model(data.x, data.edge_index)
                    cs = F.cosine_similarity(out, ftrain_fullbatch.x)
                    dr = 0
                    for i in range(NUM_NODES):
                        dr += (cs[indices[i]] < thresholds[i]).sum().item() / len(cs[indices[i]])
                    dr /= NUM_NODES

                    pbar.set_postfix_str(f"Loss: {np.mean(losses):.4f}, ADR: {dr:.4f}")
                else:
                    pbar.set_postfix_str(f"Loss: {np.mean(losses):.4f}")
        
        print(f"Average epoch time: {np.mean(epoch_times):.4f} s")

        # loading best weights
        model.load_state_dict(best_weights)
        model.eval()

        train_fullbatch = next(iter(ctrainloader)).to(DEVICE) 
        out = model(train_fullbatch.x, train_fullbatch.edge_index)

        indices = get_indices(out, NUM_NODES)
        thresholds = [0.999 * F.cosine_similarity(out[indices[i]], train_fullbatch.x[indices[i]]).min().item() for i in range(NUM_NODES)]

        @torch.no_grad()
        def detection_rate(scenario, node):
            model.eval()
            data = get_fullbatch(scenario, processed).to(DEVICE) # type: ignore
            out = model(data.x, data.edge_index)
            indices = get_indices(out, NUM_NODES)

            cs = F.cosine_similarity(out[indices[node]], data.x[indices[node]])
            return 100 * (cs < thresholds[node]).sum() / len(cs)
        

        detection_df = []
        for scenario in scenarios:
            dr = []
            for node in range(NUM_NODES):
                dr.append(detection_rate(scenario, node).float().item())
            detection_df.append(dr)

        MAIN_ADR.append(detection_df)

    # averaging performance over runs
    MAIN_ADR = np.stack(MAIN_ADR).mean(axis=0)

    from tabulate import tabulate
    if args.dataset_name == "swarm1":
        true_labels = [
            0,0,0,0,        # D1
            0,0,0,0,        # D2
            0,0,0,0,        # P1
            0,0,0,0,        # P2
            1,0,0,0,        # AN0
            0,1,1,1,        # AN1
            0,0,1,1,        # AN2
            0,0,0,1,        # AN3
            0,1,1,1,        # AN12
            0,0,1,1,        # AN23
            0,1,1,1,        # AN13
            0,1,1,1,        # AN123
            1,1,1,1         # AN0123
        ]   
    else:
        true_labels = [
            0,0,0,0,0,0,    # D1
            0,0,0,0,0,0,    # D2
            0,0,0,0,0,0,    # D3
            0,0,0,0,0,0,    # D4
            1,0,0,0,0,0,    # AN0
            0,1,1,1,0,0,    # AN1
            0,0,1,1,0,0,    # AN2
            0,0,0,1,0,0,    # AN3
            0,0,0,0,1,1,    # AN4
            0,0,0,0,0,1,    # AN5
        ]        

    true_labels = np.array(true_labels).reshape(MAIN_ADR.shape)
    out = np.zeros_like(MAIN_ADR)

    out[true_labels==0] = 100 - MAIN_ADR[true_labels==0]
    out[true_labels==1] = MAIN_ADR[true_labels==1]
    out = [list(i) for i in out]
    for n,scenario in enumerate(scenarios):
        out[n] = [scenario] + out[n]

    print(tabulate(out, headers=["Scenario"] + nodes, tablefmt="outline", floatfmt=".2f"))

    if not args.no_save:
        os.makedirs("../results/", exist_ok=True)
        os.makedirs("../models/", exist_ok=True)
        out = pd.DataFrame(out, columns=["Scenario"] + nodes)
        out.to_csv(f"../results/detection_{args.layer_type}_{args.latent}_{args.fake_mask_size}_{args.dataset_name}_{args.runs}.csv", index=False)
        torch.save(model.to('cpu'), f"../models/{args.layer_type}_model_{args.latent}_{args.fake_mask_size}_{args.dataset_name}_{args.runs}.pt")
