#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCAFFOLD Federated Learning Implementation
Based on "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" by Karimireddy et al. (2020)
"""

import torch
import copy
import numpy as np
import sys
from collections import OrderedDict

sys.path.append('../')
from utils_dir import check_gpu_pytorch

class SCAFFOLDLocalUpdate:
    """
    SCAFFOLD local update class that extends the existing LocalUpdate functionality
    """
    def __init__(self, args, dataset, idxs, logger, device=None):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))

        if device is not None:
            self.device = device
        else:
            from utils_dir import check_gpu_pytorch
            self.device = check_gpu_pytorch()
            
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        
    def train_val_test(self, dataset, idxs):
        """
        Split indexes for train, validation, and test (80, 10, 10)
        """
        from torch.utils.data import DataLoader
        from update import DatasetSplit
        
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]
        
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                batch_size=max(1, min(len(idxs_val), self.args.local_bs)), 
                                shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                               batch_size=max(1, min(len(idxs_test), self.args.local_bs)), 
                               shuffle=False)
        return trainloader, validloader, testloader

    def update_weights_scaffold(self, model, c_global, c_local, global_round):
        """
        SCAFFOLD local update with control variates (STABLE VERSION)
        """
        model.train()
        
        # Store initial model state
        initial_model_state = {name: param.clone().detach() for name, param in model.named_parameters()}
        
        # Use LOWER learning rate for stability
        lr = 0.001  # Fixed lower learning rate
        
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")
        
        epoch_losses = []
        
        for iter in range(self.args.local_ep):
            batch_losses = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                
                # Skip if loss is invalid
                if not torch.isfinite(loss):
                    continue
                    
                loss.backward()
                
                # SCAFFOLD gradient correction
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            # Apply SMALL correction to avoid explosion
                            correction = 0.01 * (c_global[name] - c_local[name])
                            param.grad.data.add_(correction)
                    
                    # AGGRESSIVE gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                batch_losses.append(loss.item())
            
            if batch_losses:
                epoch_losses.append(sum(batch_losses) / len(batch_losses))
        
        # Calculate model updates
        delta_model = OrderedDict()
        with torch.no_grad():
            for name, param in model.named_parameters():
                delta_model[name] = param.data - initial_model_state[name]
        
        # SIMPLIFIED control variate update (avoid division by small numbers)
        delta_control = OrderedDict()
        with torch.no_grad():
            for name, param in model.named_parameters():
                # Simple difference - no division
                delta_control[name] = 0.01 * (c_local[name] - c_global[name] - delta_model[name])
                
                # Clip control variates
                delta_control[name] = torch.clamp(delta_control[name], -1.0, 1.0)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
        return delta_model, delta_control, avg_loss


def initialize_control_variates(model):
    """
    Initialize control variates to zero with the same structure as model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        control_variates: OrderedDict of zero tensors matching model parameters
    """
    control_variates = OrderedDict()
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            control_variates[name] = torch.zeros_like(param.data)
    
    return control_variates


def aggregate_model_updates(delta_models):
    """
    Aggregate model updates from multiple clients
    
    Args:
        delta_models: List of model updates from clients
        
    Returns:
        aggregated_delta: Averaged model updates
    """
    aggregated_delta = OrderedDict()
    
    # Initialize with zeros
    for name in delta_models[0].keys():
        aggregated_delta[name] = torch.zeros_like(delta_models[0][name])
    
    # Sum all updates
    for delta in delta_models:
        for name, param_delta in delta.items():
            aggregated_delta[name].add_(param_delta)
    
    # Average
    num_clients = len(delta_models)
    for name in aggregated_delta.keys():
        aggregated_delta[name].div_(num_clients)
    
    return aggregated_delta


def aggregate_control_updates(delta_controls, num_total_clients):
    """
    Aggregate control variate updates from participating clients
    
    Args:
        delta_controls: List of control variate updates from participating clients
        num_total_clients: Total number of clients (N)
        
    Returns:
        aggregated_delta_c: Aggregated control variate updates
    """
    aggregated_delta_c = OrderedDict()
    
    # Initialize with zeros
    for name in delta_controls[0].keys():
        aggregated_delta_c[name] = torch.zeros_like(delta_controls[0][name])
    
    # Sum all control updates
    for delta_c in delta_controls:
        for name, control_delta in delta_c.items():
            aggregated_delta_c[name].add_(control_delta)
    
    # Scale by |S|/N where |S| is number of participating clients
    num_participating = len(delta_controls)
    scaling_factor = num_participating / num_total_clients
    
    for name in aggregated_delta_c.keys():
        aggregated_delta_c[name].mul_(scaling_factor)
    
    return aggregated_delta_c


def update_global_model(model, aggregated_delta):
    """
    Update global model with aggregated updates
    
    Args:
        model: Global model
        aggregated_delta: Aggregated model updates
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data.add_(aggregated_delta[name])


def update_control_variates(control_variates, aggregated_delta_c):
    """
    Update control variates with aggregated updates
    
    Args:
        control_variates: Current control variates
        aggregated_delta_c: Aggregated control variate updates
    """
    with torch.no_grad():
        for name in control_variates.keys():
            control_variates[name].add_(aggregated_delta_c[name])
