import os
import copy
import time
import numpy as np
from tqdm import tqdm
import torch
import sys

sys.path.append('../')
from options import args_parser
from models import CNNCifar, CNNFemnist, CNNCifar100
from utils_dir import get_dataset, exp_details , check_gpu_pytorch
from update import LocalUpdate, test_inference


class FedProxLocalUpdate(LocalUpdate):
    def __init__(self, args, dataset, idxs, logger, mu=0.01):
        super().__init__(args, dataset, idxs, logger)
        self.mu = mu
        self.optimizer = None  # Initialize optimizer later with model params
        
    def update_weights(self, model, global_round):
        # Use parent's method to ensure proper device and criterion setup
        self._ensure_device_and_criterion(model)
        
        # Initialize optimizer only once or when model changes
        if self.optimizer is None:
            if self.args.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, 
                                               momentum=self.args.momentum)
            elif self.args.optimizer == 'adam':
                self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                                weight_decay=1e-4)
        else:
            # Update optimizer's parameters if model changed
            self.optimizer.param_groups[0]['params'] = list(model.parameters())

        model.train()
        epoch_loss = []
        global_weights = copy.deepcopy(model.state_dict())  # Key for FedProx
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                
                # FedProx Proximal Term
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    proximal_term += (param - global_weights[name]).norm(2) ** 2
                loss = loss + (self.mu / 2) * proximal_term
                
                loss.backward()
                
                # Preserve gradient clipping from parent class if needed
                if self.args.dataset == 'cifar100':
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    args.mu = getattr(args, 'mu', 0.01)
    
    device = check_gpu_pytorch()

    # Load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    # Build model
    if args.model == 'cnn':
        if args.dataset == 'cifar' or args.dataset == 'cifar10':
            args.num_classes = 10
            args.num_channels = 3
            global_model = CNNCifar(args)
            print(f"CIFAR-10: Model created with {global_model.fc2.out_features} output classes")
            
        elif args.dataset == 'cifar100':
            args.num_classes = 100
            args.num_channels = 3
            global_model = CNNCifar100(args)
            print(f"CIFAR-100: Model created with {global_model.fc2.out_features} output classes")
            
        elif args.dataset == 'femnist':
            args.num_classes = 62
            args.num_channels = 1
            global_model = CNNFemnist(args)
            print(f"FEMNIST: Model created with {global_model.fc2.out_features} output classes")
            
        else:
            exit(f'Error: unsupported dataset {args.dataset}. Supported: cifar, cifar10, cifar100, femnist')
            
    else:
        exit('Error: only CNN model is supported. Use --model=cnn')
    
    global_model.to(device)
    global_model.train()
    
    train_loss, train_accuracy, test_accuracy = [], [], []
    
    for epoch in tqdm(range(args.epochs)):
        print(f'Round {epoch+1}/{args.epochs}')
        
        local_weights, local_losses = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            local_model = FedProxLocalUpdate(args=args, dataset=train_dataset,
                                           idxs=user_groups[idx], logger=None, mu=args.mu)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        
        # FedAvg aggregation
        global_weights = {}
        for key in local_weights[0].keys():
            global_weights[key] = torch.stack([local_weights[i][key] for i in range(len(local_weights))], 0).mean(0)
        
        global_model.load_state_dict(global_weights)
        
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        # Calculate train accuracy on training set (sample)
        train_acc, _ = test_inference(args, global_model, train_dataset)
        train_accuracy.append(train_acc)
        
        # Calculate test accuracy to evaluate the global model
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_accuracy.append(test_acc)
        
        # Print progress every round with BOTH train and test accuracy
        if epoch % 1 == 0:
            print(f"Round {epoch+1}: Train Accuracy = {train_acc*100:.2f}%, Test Accuracy = {test_acc*100:.2f}%, Loss = {loss_avg:.4f}")
    
    final_test_acc, _ = test_inference(args, global_model, test_dataset)
    print(f"Final Test Accuracy: {final_test_acc*100:.2f}%")
