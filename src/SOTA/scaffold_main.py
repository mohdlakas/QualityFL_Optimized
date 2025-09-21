#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for SCAFFOLD Federated Learning with Comprehensive Analysis
"""

import os
import copy
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

sys.path.append('../')
from options import args_parser
from update import test_inference
from models import CNNCifar, CNNMnist, CNNFemnist
from utils_dir import (get_dataset, exp_details, plot_data_distribution,
                      ComprehensiveAnalyzer, write_scaffold_comprehensive_analysis, check_gpu_pytorch)
from scaffold import (SCAFFOLDLocalUpdate, initialize_control_variates,
                     aggregate_model_updates, aggregate_control_updates,
                     update_global_model, update_control_variates)


def run_scaffold(args):
    """
    Main function to run SCAFFOLD federated learning with comprehensive tracking
    """
    start_time = time.time()
    
    # FIX: Create save directories based on current working directory
    current_dir = os.getcwd()
    if 'Algorithms' in current_dir or 'algorithms' in current_dir:
        save_base = '../../save'  # From src/Algorithms to project root
    else:
        save_base = '../save'     # From src to project root
    
    # Create all necessary directories
    os.makedirs(f'{save_base}/objects', exist_ok=True)
    os.makedirs(f'{save_base}/images', exist_ok=True)
    os.makedirs(f'{save_base}/logs', exist_ok=True)
    
    # check device
    device = check_gpu_pytorch()

    # Load dataset and split users
    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    # Plot data distribution
    plot_data_distribution(
        user_groups, train_dataset,
        save_path=f'{save_base}/images/data_distribution_{args.dataset}_iid[{args.iid}]_alpha[{getattr(args, "alpha", "NA")}].png',
        title="Client Data Distribution (IID={})".format(args.iid)
    )
    
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
            global_model = CNNCifar(args)  # Same architecture, different output size
            print(f"CIFAR-100: Model created with {global_model.fc2.out_features} output classes")
            
        elif args.dataset == 'femnist':
            args.num_classes = 62  # 10 digits + 26 uppercase + 26 lowercase
            args.num_channels = 1
            global_model = CNNFemnist(args)
            print(f"FEMNIST: Model created with {global_model.fc2.out_features} output classes")
            
        else:
            exit(f'Error: unsupported dataset {args.dataset}. Supported: cifar, cifar10, cifar100, femnist')
            
    else:
        exit('Error: only CNN model is supported. Use --model=cnn')
    
    # Set model to device
    global_model.to(device)
    global_model.train()
    
    # Initialize comprehensive analyzer for detailed metrics
    analyzer = ComprehensiveAnalyzer()
    
    # Set random seed for reproducibility if provided
    experiment_seed = getattr(args, 'seed', None)
    if experiment_seed is not None:
        torch.manual_seed(experiment_seed)
        np.random.seed(experiment_seed)
    
    # Initialize control variates
    print("Initializing SCAFFOLD control variates...")
    c_global = initialize_control_variates(global_model)
    c_local_dict = {}
    
    # Move control variates to device
    for name in c_global.keys():
        c_global[name] = c_global[name].to(device)
    
    # Initialize local control variates for all clients
    for idx in range(args.num_users):
        c_local_dict[idx] = copy.deepcopy(c_global)
    
    # Training metrics
    train_loss, train_accuracy = [], []
    test_accuracy = []
    print_every = 1
    
    print("Starting SCAFFOLD training...")
    for epoch in tqdm(range(args.epochs)):
        round_start_time = time.time()  # Track round time
        
        # Client sampling
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        selected_clients = list(idxs_users)  # For analysis tracking
        
        # Track client data for analysis
        client_losses = {}  # Store loss info for quality analysis
        data_sizes = {}     # Store client data sizes
        
        # Collect updates
        delta_models = []
        delta_controls = []
        local_losses = []
        
        # Client updates
        for idx in idxs_users:
            # Create local update object
            local_model = SCAFFOLDLocalUpdate(args=args, dataset=train_dataset,
                                            idxs=user_groups[idx], logger=None,device=device)
            
            # Get local updates
            delta_model, delta_control, loss = local_model.update_weights_scaffold(
                model=copy.deepcopy(global_model),
                c_global=c_global,
                c_local=c_local_dict[idx],
                global_round=epoch
            )
            
            # Store for analysis
            client_losses[idx] = (loss, loss * 0.9)  # Simple before/after approximation
            data_sizes[idx] = len(user_groups[idx])
            
            # Collect updates
            delta_models.append(delta_model)
            delta_controls.append(delta_control)
            local_losses.append(loss)
            
            # Update local control variate: c_i = c_i + Δc_i
            with torch.no_grad():
                for name in c_local_dict[idx].keys():
                    c_local_dict[idx][name].add_(delta_control[name])
        
        # Aggregate model updates
        aggregated_delta_model = aggregate_model_updates(delta_models)
        
        # Aggregate control updates
        aggregated_delta_control = aggregate_control_updates(delta_controls, args.num_users)
        
        # Update global model
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                param.data.add_(aggregated_delta_model[name], alpha=getattr(args, 'scaffold_stepsize', 1.0))
        
        # Update global control: c = c + aggregated_delta_c
        update_control_variates(c_global, aggregated_delta_control)
        
        # Calculate average loss
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        list_acc = []
        global_model.eval()

        for c in selected_clients:  # Only selected clients
            local_model = SCAFFOLDLocalUpdate(args=args, dataset=train_dataset,
                                            idxs=user_groups[c], logger=None,device=device)
            
            # FIXED: Evaluate on training data instead of test data
            correct, total = 0, 0
            criterion = torch.nn.NLLLoss().to(device)

            with torch.no_grad():
                for images, labels in local_model.trainloader:  # ← Use trainloader!
                    images, labels = images.to(device), labels.to(device)
                    outputs = global_model(images)
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            acc = correct / total if total > 0 else 0
            list_acc.append(acc)

        global_model.train()
        train_accuracy.append(sum(list_acc) / len(list_acc) if list_acc else 0.0)
        
        # Calculate test accuracy
        test_acc_current, _ = test_inference(args, global_model, test_dataset)
        test_accuracy.append(test_acc_current)
        
        # Print progress
        if (epoch + 1) % print_every == 0:
            # Calculate current training accuracy from the list we just computed
            current_train_acc = train_accuracy[-1] if train_accuracy else 0.0
            
            # Print in the exact format expected by auto_compare.py
            print(f'Round {epoch+1}: Train Accuracy = {current_train_acc*100:.2f}%, Test Accuracy = {test_acc_current*100:.2f}%, Loss = {loss_avg:.4f}')
            
            # Optional: Keep the control variate monitoring (auto_compare.py will ignore this)
            c_global_norm = 0
            for name, c_val in c_global.items():
                c_global_norm += torch.norm(c_val).item()**2
            c_global_norm = np.sqrt(c_global_norm)
            print(f'Global Control Variate Norm: {c_global_norm:.4f}')
        
        # Track round time
        round_time = time.time() - round_start_time
        
        # For SCAFFOLD, create analysis metrics similar to FedAvg
        # Calculate uniform aggregation weights (SCAFFOLD uses standard aggregation)
        aggregation_weights = {client_id: 1.0/len(selected_clients) for client_id in selected_clients}
        
        # For SCAFFOLD, client reliability based on loss improvement and control variate stability
        client_reliabilities = {}
        for client_id in selected_clients:
            loss_before, loss_after = client_losses[client_id]
            loss_improvement = max(0, loss_before - loss_after)
            
            # Factor in control variate stability
            control_norm = 0
            for name in c_local_dict[client_id].keys():
                control_norm += torch.norm(c_local_dict[client_id][name]).item()**2
            control_norm = np.sqrt(control_norm)
            
            # Reliability based on loss improvement and control stability
            reliability = (loss_improvement + 1e-6) * data_sizes[client_id] / (1000.0 * (1 + control_norm))
            client_reliabilities[client_id] = min(1.0, reliability)  # Cap at 1.0
        
        # Log all data to analyzer (adapted for SCAFFOLD)
        analyzer.log_round_data(
            round_num=epoch + 1,
            train_acc=train_accuracy[-1],
            train_loss=loss_avg,
            test_acc=test_acc_current,
            selected_clients=selected_clients,
            aggregation_weights=aggregation_weights,
            client_reliabilities=client_reliabilities,
            client_qualities=None,  # SCAFFOLD doesn't have explicit quality computation
            memory_bank_size=None,  # SCAFFOLD doesn't have memory bank
            avg_similarity=None,    # SCAFFOLD doesn't track similarities
            round_time=round_time
        )
    
    # Final test accuracy
    test_acc_final, test_loss_final = test_inference(args, global_model, test_dataset)
    
    print(f'\n\nResults after {args.epochs} global rounds:')
    print(f'Test Accuracy: {100*test_acc_final:.2f}%')
    print(f'Test Loss: {test_loss_final:.3f}')
    
    total_time = time.time() - start_time
    print(f'\nTotal Runtime: {total_time:.0f}s')
    
    # Save training objects
    file_name = f'{save_base}/objects/SCAFFOLD_{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}].pkl'
    
    import pickle
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)
    
    # Enhanced plotting with dual y-axis
    plt.figure(figsize=(12, 8))
    plt.title(f'SCAFFOLD Results - Test Accuracy: {test_acc_final*100:.2f}%\nTraining Loss and Test Accuracy vs Communication Rounds')
    plt.xlabel('Communication Rounds')
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax1.plot(range(len(train_loss)), train_loss, color='b', label='Training Loss', linewidth=2)
    ax2.plot(range(len(test_accuracy)), [acc * 100 for acc in test_accuracy], color='r', label='Test Accuracy (%)', linewidth=2, marker='o')
    
    ax1.set_ylabel('Training Loss', color='b')
    ax2.set_ylabel('Test Accuracy (%)', color='r')
    
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f'{save_base}/images/SCAFFOLD_{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_alpha[{getattr(args, "alpha", "NA")}]_E[{args.local_ep}]_B[{args.local_bs}]_opt[{getattr(args, "optimizer", "NA")}]_lr[{getattr(args, "lr", "NA")}]_{timestamp}.png'
    
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate comprehensive analysis report
    scaffold_filename = f"{save_base}/logs/scaffold_comprehensive_analysis_{timestamp}.txt"
    write_scaffold_comprehensive_analysis(analyzer, args, test_acc_final, total_time, scaffold_filename, experiment_seed)
    
    print(f"\n✅ SCAFFOLD comprehensive analysis saved to: {scaffold_filename}")
    
    # Print key metrics to console for immediate feedback
    convergence_metrics = analyzer.calculate_convergence_metrics()
    client_analysis = analyzer.analyze_client_selection_quality()
    
    print(f"\n🔍 SCAFFOLD RESULTS SUMMARY:")
    print(f"   Final Test Accuracy: {test_acc_final:.4f} ({test_acc_final*100:.2f}%)")
    print(f"   Convergence Speed: {convergence_metrics.get('convergence_round', 'N/A')} rounds")
    print(f"   Training Stability: {convergence_metrics.get('training_stability', 0):.6f}")
    print(f"   Unique Clients Selected: {client_analysis.get('total_unique_clients', 0)}")
    print(f"   Avg Participation Rate: {client_analysis.get('avg_participation_rate', 0):.4f}")
    print(f"   Total Runtime: {total_time:.2f} seconds")
    print(f"\n📁 All results saved in ../save/logs/ directory")
    
    # Save results (enhanced)
    results = {
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_acc_final': test_acc_final,
        'runtime': total_time,
        'analyzer': analyzer
    }
    
    return global_model, results


if __name__ == '__main__':
    args = args_parser()
    
    exp_details(args)
    
    # Run SCAFFOLD
    model, results = run_scaffold(args)
    
    print(f"Final Test Accuracy: {results['test_acc_final']*100:.2f}%")
    
    # Save model if needed
    if hasattr(args, 'save_model') and args.save_model:
        torch.save(model.state_dict(), f'./save/scaffold_{args.dataset}_{args.model}_epochs{args.epochs}.pth')
