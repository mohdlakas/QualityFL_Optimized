from datetime import datetime
import copy
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys
import json
from scipy import stats
from collections import defaultdict, deque

# Modern Dirichlet-based imports
from sampling_dir import (cifar10_iid, cifar10_dirichlet, cifar100_iid, cifar100_dirichlet,
                     mnist_iid, mnist_dirichlet, analyze_distribution, plot_distribution)

# Legacy imports (commented out - use modern Dirichlet methods instead)
# from sampling import mnist_noniid, mnist_noniid_unequal, cifar_noniid

def set_seed(seed, enable_deterministic=True):
    """
    Set all random seeds for reproducibility across the entire codebase.
    
    Args:
        seed: Random seed value
        enable_deterministic: Enable deterministic behavior for CUDA/cuDNN
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # For complete determinism (optional)
    if enable_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Python hash randomization
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # CUDA determinism
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

def check_gpu_pytorch():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        
        print(f"GPU is available!")
        print(f"Number of GPUs: {gpu_count}")
        print(f"GPU Name: {gpu_name}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        
        return device
    else:
        print("GPU is not available, using CPU")
        return torch.device("cpu")
    
def get_dataset(args):
    """ 
    Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    
    Now uses modern Dirichlet distribution for non-IID data instead of legacy shard methods.
    """

    if args.dataset == 'cifar' or args.dataset == 'cifar10' or args.dataset == 'cifar100':
        # Determine which CIFAR dataset to use
        if args.dataset == 'cifar' or args.dataset == 'cifar10':
            dataset_class = datasets.CIFAR10
            data_dir = './data/cifar/'
            args.num_classes = 10
            dataset_name = 'CIFAR-10'
        else:  # cifar100
            dataset_class = datasets.CIFAR100
            data_dir = './data/cifar100/'
            args.num_classes = 100
            dataset_name = 'CIFAR-100'

        args.num_channels = 3  # Auto-update for CIFAR datasets

        # Different transforms for CIFAR-10 vs CIFAR-100
        if args.dataset == 'cifar100':
            # CIFAR-100 optimized transforms
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])
            
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])
        else:
            # CIFAR-10 transforms
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            test_transform = train_transform

        train_dataset = dataset_class(data_dir, train=True, download=True,
                                     transform=train_transform)
        test_dataset = dataset_class(data_dir, train=False, download=True,
                                    transform=test_transform)

        # Sample training data amongst users
        sampling_seed = getattr(args, 'seed', None)
        if args.iid:
            print(f"Creating IID data distribution for {dataset_name}...")
            if args.dataset == 'cifar' or args.dataset == 'cifar10':
                user_groups = cifar10_iid(train_dataset, args.num_users, seed=sampling_seed)
            else:
                user_groups = cifar100_iid(train_dataset, args.num_users, seed=sampling_seed)
        else:
            # Non-IID distribution
            alpha = getattr(args, 'alpha', 0.5)
            min_samples = getattr(args, 'min_samples', 50)

            print(f"Creating Non-IID data distribution for {dataset_name} with alpha={alpha}...")
            
            if args.dataset == 'cifar' or args.dataset == 'cifar10':
                user_groups = cifar10_dirichlet(train_dataset, args.num_users, 
                                              alpha=alpha, min_samples_per_user=min_samples, seed=sampling_seed)
            else:
                # Check if enhanced CIFAR-100 function exists, otherwise use standard
                try:
                    from sampling_dir import cifar100_enhanced_noniid
                    user_groups = cifar100_enhanced_noniid(train_dataset, args.num_users, 
                                                         alpha=alpha, classes_per_user=10)
                    print("Using enhanced CIFAR-100 non-IID distribution")
                except ImportError:
                    # Fallback to standard Dirichlet
                    user_groups = cifar100_dirichlet(train_dataset, args.num_users, 
                                                   alpha=alpha, min_samples_per_user=min_samples, seed=sampling_seed)
                    print("Using standard Dirichlet distribution for CIFAR-100")

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.dataset == 'mnist':
            data_dir = './data/mnist/'
            dataset_class = datasets.MNIST
        else:
            data_dir = './data/fmnist/'
            dataset_class = datasets.FashionMNIST

        args.num_classes = 10
        args.num_channels = 1

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = dataset_class(data_dir, train=True, download=True,
                                     transform=apply_transform)
        test_dataset = dataset_class(data_dir, train=False, download=True,
                                    transform=apply_transform)

        # Sample training data amongst users
        sampling_seed = getattr(args, 'seed', None)
        if args.iid:
            print(f"Creating IID data distribution for {args.dataset.upper()}...")
            user_groups = mnist_iid(train_dataset, args.num_users, seed=sampling_seed)
        else:
            alpha = getattr(args, 'alpha', 0.5)
            min_samples = getattr(args, 'min_samples', 50)
            
            print(f"Creating Non-IID data distribution for {args.dataset.upper()} with alpha={alpha}...")
            user_groups = mnist_dirichlet(train_dataset, args.num_users, 
                                        alpha=alpha, min_samples_per_user=min_samples, seed=sampling_seed)

    elif args.dataset == 'femnist':
        # Use the FEMNIST data from the project data directory
        data_dir = './data/femnist/data'  # Changed from './leaf/data/femnist/data'
        args.num_classes = 62  # 10 digits + 26 uppercase + 26 lowercase
        args.num_channels = 1
        
        # Load FEMNIST data from train and test directories
        train_data_dir = os.path.join(data_dir, 'train')
        test_data_dir = os.path.join(data_dir, 'test')
        
        # Check if directories exist
        if not os.path.exists(train_data_dir) or not os.path.exists(test_data_dir):
            print(f"❌ FEMNIST data not found at {data_dir}")
            print(f"   Train dir exists: {os.path.exists(train_data_dir)}")
            print(f"   Test dir exists: {os.path.exists(test_data_dir)}")
            raise FileNotFoundError(f"FEMNIST data not found at {data_dir}")
        
        # Load training data
        user_groups = {}
        train_images, train_labels = [], []
        test_images, test_labels = [], []
        
        print(f"📖 Loading FEMNIST training data from {train_data_dir}...")
        train_files = [f for f in os.listdir(train_data_dir) if f.endswith('.json')]
        train_files.sort()
        
        if len(train_files) == 0:
            raise FileNotFoundError("No FEMNIST train files found")
        
        user_id = 0
        for file in train_files:
            file_path = os.path.join(train_data_dir, file)
            file_size = os.path.getsize(file_path) / (1024**3)
            print(f"   Processing {file} ({file_size:.2f} GB)...")
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                print(f"   Found {len(data['users'])} users in {file}")
                
                for user in data['users']:
                    user_data = data['user_data'][user]
                    start_idx = len(train_images)
                    
                    # Convert flat image data to 28x28
                    user_images = np.array(user_data['x']).reshape(-1, 28, 28)
                    user_labels = np.array(user_data['y'])
                    
                    train_images.extend(user_images)
                    train_labels.extend(user_labels)
                    
                    end_idx = len(train_images)
                    user_groups[user_id] = list(range(start_idx, end_idx))
                    user_id += 1
                    
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
                raise
        
        print(f"✅ Loaded {user_id} users from training data")
        
        print(f"📖 Loading FEMNIST test data from {test_data_dir}...")
        test_files = [f for f in os.listdir(test_data_dir) if f.endswith('.json')]
        test_files.sort()
        
        for file in test_files:
            file_path = os.path.join(test_data_dir, file)
            file_size = os.path.getsize(file_path) / (1024**3)
            print(f"   Processing {file} ({file_size:.2f} GB)...")
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                for user in data['users']:
                    user_data = data['user_data'][user]
                    user_images = np.array(user_data['x']).reshape(-1, 28, 28)
                    user_labels = np.array(user_data['y'])
                    
                    test_images.extend(user_images)
                    test_labels.extend(user_labels)
                    
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
                raise
        
        # Convert to numpy arrays and normalize
        train_images = np.array(train_images).astype(np.float32) / 255.0
        train_labels = np.array(train_labels)
        test_images = np.array(test_images).astype(np.float32) / 255.0
        test_labels = np.array(test_labels)
        
        print(f"📊 Final data: {len(train_images)} train, {len(test_images)} test samples")
        print(f"📊 Users: {len(user_groups)}, Classes: {len(np.unique(train_labels))}")
        
        # Create transforms for FEMNIST
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.9637,), (0.1597,))  # FEMNIST specific normalization
        ])
        
        class FEMNISTDataset(torch.utils.data.Dataset):
            def __init__(self, images, labels, transform=None):
                self.images = images  # Already in [0, 1] range
                self.labels = labels
                self.targets = labels  # For compatibility
                self.transform = transform
                
            def __len__(self):
                return len(self.images)
                
            def __getitem__(self, idx):
                image = self.images[idx]  # Already float32 in [0, 1]
                label = self.labels[idx]
                
                if self.transform:
                    # Option 1: If transform expects PIL Image
                    # image_pil = Image.fromarray((image * 255).astype(np.uint8))
                    # image = self.transform(image_pil)
                    
                    # Option 2: Direct tensor transform (more efficient)
                    image = torch.FloatTensor(image).unsqueeze(0)  # Add channel dim
                    # Apply normalization directly
                    image = (image - 0.9637) / 0.1597
                else:
                    image = torch.FloatTensor(image).unsqueeze(0)
                    
                return image, label
        
        # Simplified transform - no need for ToPILImage
        train_dataset = FEMNISTDataset(train_images, train_labels, transform=True)
        test_dataset = FEMNISTDataset(test_images, test_labels, transform=True)
        
        print(f"✅ FEMNIST dataset loaded: {len(user_groups)} users, {len(train_dataset)} train samples, {len(test_dataset)} test samples")

    else:
        raise ValueError(f"Dataset '{args.dataset}' not supported. Choose from: 'mnist', 'fmnist', 'cifar', 'cifar10', 'cifar100', 'femnist'")

    # Optional: Analyze data distribution if requested
    if hasattr(args, 'analyze_data') and args.analyze_data:
        dataset_name = args.dataset.upper()
        iid_status = "IID" if args.iid else f"Non-IID (α={getattr(args, 'alpha', 0.5)})"
        analyze_distribution(user_groups, train_dataset, f"{dataset_name} {iid_status}")

    # Optional: Plot data distribution if requested
    if hasattr(args, 'plot_data') and args.plot_data:
        dataset_name = args.dataset.upper()
        iid_status = "IID" if args.iid else f"Non-IID (α={getattr(args, 'alpha', 0.5)})"
        save_path = getattr(args, 'save_plot', None)
        plot_distribution(user_groups, train_dataset, f"{dataset_name} {iid_status}", save_path)

    return train_dataset, test_dataset, user_groups

def evaluate_global_model_on_training_data(model, train_dataset, user_groups, device, batch_size=64):
    """Evaluate global model on all training data."""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    # Combine all client data
    all_indices = []
    for client_indices in user_groups.values():
        all_indices.extend(client_indices)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, all_indices),
        batch_size=batch_size, shuffle=False
    )
    
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    
    return 100.0 * total_correct / total_samples


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg




def exp_details(args):
    """
    Print experiment details with modern Dirichlet distribution information
    """
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}')
    print(f'    Dataset   : {args.dataset.upper()}')
    print(f'    Classes   : {args.num_classes}')
    print(f'    Channels  : {args.num_channels}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    Distribution: IID')
    else:
        print('    Distribution: Non-IID (Dirichlet)')
        alpha = getattr(args, 'alpha', 0.5)
        print(f'    Dirichlet Alpha: {alpha}')
        
        # Provide interpretation of alpha value
        if alpha >= 10.0:
            print('    Heterogeneity: Nearly IID')
        elif alpha >= 1.0:
            print('    Heterogeneity: Moderate')
        elif alpha >= 0.5:
            print('    Heterogeneity: High')
        elif alpha >= 0.1:
            print('    Heterogeneity: Extreme')
        else:
            print('    Heterogeneity: Very Extreme')
            
        min_samples = getattr(args, 'min_samples', 10)
        print(f'    Min samples per user: {min_samples}')
    
    print(f'    Total users: {args.num_users}')
    print(f'    Fraction of users: {args.frac}')
    print(f'    Local Batch size: {args.local_bs}')
    print(f'    Local Epochs: {args.local_ep}\n')

    # Print PUMB-specific parameters if present
    if hasattr(args, 'pumb_exploration_ratio'):
        print(f'    PUMB Exploration Ratio: {args.pumb_exploration_ratio}')
    if hasattr(args, 'pumb_initial_rounds'):
        print(f'    PUMB Initial Rounds: {args.pumb_initial_rounds}')

    # Quality metric parameters (dynamic version)
    print('\n    Quality Metric Parameters:')
    if hasattr(args, 'quality_alpha') and hasattr(args, 'quality_metric_type'):
        print(f'    Metric Type: {args.quality_metric_type}')
        print(f'    Alpha (Loss Weight): {args.quality_alpha}')
        print(f'    Beta (Consistency Weight): {args.quality_beta}') 
        print(f'    Gamma (Data Weight): {args.quality_gamma}')
        
        # ADD: Display baseline quality if available
        if hasattr(args, 'quality_baseline'):
            print(f'    Baseline Quality: {args.quality_baseline}')

    else:
        # Fallback to default values (FedAvg doesn't use quality metrics)
        print(f'    Quality Metric: Not applicable')
        
    print()


def calculate_statistics(values):
    """Calculate comprehensive statistics for a sequence of values."""
    if not values:
        return {
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'median': 0,
            'count': 0,
            'trend': 0
        }
    
    values_array = np.array(values)
    
    # Calculate trend (simple linear regression slope)
    x = np.arange(len(values_array))
    if len(values_array) > 1:
        trend = np.polyfit(x, values_array, 1)[0]
    else:
        trend = 0
    
    return {
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'median': float(np.median(values_array)),
        'count': len(values),
        'trend': float(trend)
    }

def extract_parameter_statistics(model_params):
    """Extract comprehensive statistics from model parameters."""
    if isinstance(model_params, dict):
        # State dict
        all_values = torch.cat([p.flatten() for p in model_params.values()])
    elif isinstance(model_params, list):
        # List of tensors
        all_values = torch.cat([p.flatten() for p in model_params])
    elif isinstance(model_params, torch.Tensor):
        # Single tensor
        all_values = model_params.flatten()
    else:
        # Unsupported type
        raise TypeError("Unsupported parameter type")
    
    # Convert to numpy
    all_values = all_values.detach().cpu().numpy()
    
    # Calculate statistics
    return {
        'mean': float(np.mean(all_values)),
        'std': float(np.std(all_values)),
        'min': float(np.min(all_values)),
        'max': float(np.max(all_values)),
        'median': float(np.median(all_values)),
        'l1_norm': float(np.linalg.norm(all_values, 1)),
        'l2_norm': float(np.linalg.norm(all_values, 2)),
        'non_zeros': float((np.abs(all_values) > 1e-6).sum() / all_values.size)
    }
# def plot_data_distribution(user_groups, dataset, save_path=None, title="Client Data Distribution"):
#     labels = np.array(dataset.targets)
#     num_clients = len(user_groups)
#     num_classes = len(np.unique(labels))
#     dist_matrix = np.zeros((num_clients, num_classes), dtype=int)
#     for user, idxs in user_groups.items():
#         user_labels = labels[list(idxs)]
#         for c in range(num_classes):
#             dist_matrix[user, c] = np.sum(user_labels == c)
#     plt.figure(figsize=(12, 6))
#     plt.imshow(dist_matrix, aspect='auto', cmap='viridis')
#     plt.colorbar(label='Samples per class')
#     plt.xlabel('Class')
#     plt.ylabel('Client')
#     plt.title(title)
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#     else:
#         plt.show()
#     plt.close()
def plot_data_distribution(user_groups, dataset, save_path=None, title="Client Data Distribution"):
    # Handle different ways to get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # Fallback: extract labels by iterating through dataset
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        labels = np.array(labels)
    
    num_clients = len(user_groups)
    num_classes = len(np.unique(labels))
    dist_matrix = np.zeros((num_clients, num_classes), dtype=int)
    
    for user, idxs in user_groups.items():
        user_labels = labels[list(idxs)]
        for c in range(num_classes):
            dist_matrix[user, c] = np.sum(user_labels == c)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(dist_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='Samples per class')
    plt.xlabel('Class')
    plt.ylabel('Client')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def exp_details_to_file(args, filename):
    """Write experiment details to a file using exp_details()."""
    with open(filename, "w") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        exp_details(args)
        sys.stdout = old_stdout


class ComprehensiveAnalyzer:
    """
    Comprehensive analysis class for PUMB federated learning experiments.
    Tracks all metrics needed for research paper validation.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracking variables."""
        # Performance metrics
        self.train_accuracies = []
        self.test_accuracies = []
        self.train_losses = []
        self.round_times = []
        
        # Client selection tracking
        self.client_selections = defaultdict(list)  # client_id -> [round1, round2, ...]
        self.client_reliability_history = defaultdict(list)  # client_id -> [reliability1, ...]
        self.client_quality_scores = defaultdict(list)  # client_id -> [quality1, ...]
        self.aggregation_weights_history = []  # [round1_weights_dict, round2_weights_dict, ...]
        
        # Memory bank tracking
        self.memory_bank_sizes = []
        self.similarity_scores = []
        self.client_ranking_history = []  # [round1_rankings, round2_rankings, ...]
        
        # Convergence tracking
        self.convergence_round = None
        self.final_accuracy = None

            # 🆕 TRACK ACTUAL VALUES USED
        self.actual_config = {
            'pumb_exploration_ratio': None,
            'pumb_initial_rounds': None,
            'quality_metric_type': None,
            'memory_bank_size_limit': None,
            'embedding_dim': None
        }

    def track_actual_config(self, **config_values):
        """Track the actual configuration values used during execution."""
        self.actual_config.update(config_values)
        print(f"📊 Tracking actual config: {config_values}")

    def log_round_data(self, round_num, train_acc, train_loss, test_acc=None, 
                      selected_clients=None, aggregation_weights=None, 
                      client_reliabilities=None, client_qualities=None,
                      memory_bank_size=None, avg_similarity=None, round_time=None):
        """Log data for a single round."""
        
        # Performance metrics
        self.train_accuracies.append(train_acc)
        self.train_losses.append(train_loss)
        if test_acc is not None:
            self.test_accuracies.append((round_num, test_acc))
        if round_time is not None:
            self.round_times.append(round_time)
            
        # Client selection tracking
        if selected_clients:
            for client_id in selected_clients:
                self.client_selections[client_id].append(round_num)
                
        if aggregation_weights:
            self.aggregation_weights_history.append(aggregation_weights.copy())
            
        if client_reliabilities:
            for client_id, reliability in client_reliabilities.items():
                self.client_reliability_history[client_id].append(reliability)
                
        if client_qualities:
            for client_id, quality in client_qualities.items():
                self.client_quality_scores[client_id].append(quality)
                
        # Memory bank tracking
        if memory_bank_size is not None:
            self.memory_bank_sizes.append(memory_bank_size)
            
        if avg_similarity is not None:
            self.similarity_scores.append(avg_similarity)
            
        # Track client rankings (reliability-based)
        if client_reliabilities:
            rankings = sorted(client_reliabilities.items(), key=lambda x: x[1], reverse=True)
            self.client_ranking_history.append(rankings)
    
    def track_aggregation_weights(self, round_num, weights):
        """Track aggregation weights for each round."""
        if not hasattr(self, 'aggregation_weights_history'):
            self.aggregation_weights_history = []
        
        # Store a copy of the weights for this round
        self.aggregation_weights_history.append(weights.copy())
        
        # Also track weight statistics
        if weights:
            weight_values = list(weights.values())
            weight_std = np.std(weight_values)
            weight_mean = np.mean(weight_values)
            weight_max = max(weight_values)
            weight_min = min(weight_values)
            
            print(f"Round {round_num} weight stats: mean={weight_mean:.3f}, std={weight_std:.3f}, "
                f"min={weight_min:.3f}, max={weight_max:.3f}")

    def track_round_time(self, round_time):
        """Track time taken for each round."""
        if not hasattr(self, 'round_times'):
            self.round_times = []
        self.round_times.append(round_time)

    def track_client_selection(self, selected_clients):
        """Track client selection for each round."""
        if not hasattr(self, 'client_selections'):
            self.client_selections = defaultdict(list)
        
        for client_id in selected_clients:
            self.client_selections[client_id].append(len(self.train_accuracies))

    def track_memory_bank_size(self, size):
        """Track memory bank size growth."""
        if not hasattr(self, 'memory_bank_sizes'):
            self.memory_bank_sizes = []
        self.memory_bank_sizes.append(size)

    def track_client_reliability(self, client_id, reliability):
        """Track client reliability over time."""
        if not hasattr(self, 'client_reliability_history'):
            self.client_reliability_history = defaultdict(list)
        self.client_reliability_history[client_id].append(reliability)

    def track_training_accuracy(self, accuracy):
        """Track training accuracy."""
        if not hasattr(self, 'train_accuracies'):
            self.train_accuracies = []
        self.train_accuracies.append(accuracy)

    def track_test_accuracy(self, accuracy):
        """Track test accuracy."""
        if not hasattr(self, 'test_accuracies'):
            self.test_accuracies = []
        self.test_accuracies.append(accuracy)

    def calculate_convergence_metrics(self, target_ratio=0.8):
        """Calculate convergence speed and stability metrics."""
        if len(self.train_accuracies) == 0:
            return {}
            
        self.final_accuracy = self.train_accuracies[-1]
        target_accuracy = target_ratio * self.final_accuracy
        
        # Find convergence round (first round to reach target)
        for i, acc in enumerate(self.train_accuracies):
            if acc >= target_accuracy:
                self.convergence_round = i + 1
                break
                
        # Training stability (std of last 10 rounds)
        stability_window = min(10, len(self.train_accuracies))
        last_accuracies = self.train_accuracies[-stability_window:]
        training_stability = np.std(last_accuracies)
        
        return {
            'final_accuracy': self.final_accuracy,
            'convergence_round': self.convergence_round or len(self.train_accuracies),
            'training_stability': training_stability,
            'convergence_target': target_accuracy,
            'target_ratio': target_ratio
        }
    
    def analyze_client_selection_quality(self):
        """Analyze client selection patterns and quality."""
        if not self.client_selections:
            return {}
            
        # Participation frequency analysis
        total_rounds = len(self.train_accuracies)
        participation_rates = {}
        for client_id, rounds in self.client_selections.items():
            participation_rates[client_id] = len(rounds) / total_rounds
            
        # Quartile analysis
        if len(participation_rates) >= 4:
            sorted_rates = sorted(participation_rates.values())
            n = len(sorted_rates)
            q1_threshold = sorted_rates[n//4]
            q3_threshold = sorted_rates[3*n//4]
            
            top_quartile_rate = np.mean([rate for rate in sorted_rates if rate >= q3_threshold])
            bottom_quartile_rate = np.mean([rate for rate in sorted_rates if rate <= q1_threshold])
        else:
            top_quartile_rate = bottom_quartile_rate = np.mean(list(participation_rates.values()))
            
        # Reliability learning analysis
        reliability_improvement = {}
        for client_id, reliabilities in self.client_reliability_history.items():
            if len(reliabilities) >= 2:
                early_avg = np.mean(reliabilities[:len(reliabilities)//3] or reliabilities[:1])
                late_avg = np.mean(reliabilities[-len(reliabilities)//3:] or reliabilities[-1:])
                reliability_improvement[client_id] = late_avg - early_avg
                
        avg_reliability_improvement = np.mean(list(reliability_improvement.values())) if reliability_improvement else 0
        
        # Quality score distribution
        all_qualities = []
        for qualities in self.client_quality_scores.values():
            all_qualities.extend(qualities)
            
        quality_stats = {
            'mean': np.mean(all_qualities) if all_qualities else 0,
            'std': np.std(all_qualities) if all_qualities else 0,
            'min': np.min(all_qualities) if all_qualities else 0,
            'max': np.max(all_qualities) if all_qualities else 0
        }
        
        return {
            'total_unique_clients': len(self.client_selections),
            'avg_participation_rate': np.mean(list(participation_rates.values())),
            'top_quartile_participation': top_quartile_rate,
            'bottom_quartile_participation': bottom_quartile_rate,
            'participation_inequality': top_quartile_rate - bottom_quartile_rate,
            'avg_reliability_improvement': avg_reliability_improvement,
            'quality_score_stats': quality_stats,
            'clients_with_reliability_data': len(self.client_reliability_history)
        }
    
    def analyze_memory_bank_effectiveness(self):
        """Analyze memory bank growth and effectiveness."""
        if not self.memory_bank_sizes:
            return {}
            
        # Memory bank growth
        initial_size = self.memory_bank_sizes[0] if self.memory_bank_sizes else 0
        final_size = self.memory_bank_sizes[-1] if self.memory_bank_sizes else 0
        growth_rate = (final_size - initial_size) / len(self.memory_bank_sizes) if self.memory_bank_sizes else 0
        
        # Similarity analysis
        avg_similarity = np.mean(self.similarity_scores) if self.similarity_scores else 0
        similarity_trend = 0
        if len(self.similarity_scores) >= 2:
            x = np.arange(len(self.similarity_scores))
            similarity_trend = np.polyfit(x, self.similarity_scores, 1)[0]
            
        # Client ranking stability (how much rankings change over time)
        ranking_stability = 0
        if len(self.client_ranking_history) >= 2:
            stability_scores = []
            for i in range(1, len(self.client_ranking_history)):
                prev_ranks = {client: rank for rank, (client, _) in enumerate(self.client_ranking_history[i-1])}
                curr_ranks = {client: rank for rank, (client, _) in enumerate(self.client_ranking_history[i])}
                
                # Calculate rank correlation for common clients
                common_clients = set(prev_ranks.keys()) & set(curr_ranks.keys())
                if len(common_clients) >= 2:
                    prev_values = [prev_ranks[c] for c in common_clients]
                    curr_values = [curr_ranks[c] for c in common_clients]
                    corr, _ = stats.spearmanr(prev_values, curr_values)
                    stability_scores.append(corr if not np.isnan(corr) else 0)
                    
            ranking_stability = np.mean(stability_scores) if stability_scores else 0
        
        return {
            'initial_memory_size': initial_size,
            'final_memory_size': final_size,
            'memory_growth_rate': growth_rate,
            'avg_similarity_score': avg_similarity,
            'similarity_trend': similarity_trend,
            'ranking_stability': ranking_stability,
            'total_rounds_tracked': len(self.memory_bank_sizes)
        }
    

    def generate_fedavg_report(self, args, test_acc, total_time, experiment_seed=None):
        """Generate a comprehensive analysis report for FedAvg (without PUMB features)."""
        convergence_metrics = self.calculate_convergence_metrics()
        
        # Simplified client analysis (no reliability/quality for FedAvg)
        client_analysis = {}
        if self.client_selections:
            participation_rates = {}
            total_rounds = len(self.train_accuracies)
            for client_id, rounds in self.client_selections.items():
                participation_rates[client_id] = len(rounds) / total_rounds
                
            client_analysis = {
                'total_unique_clients': len(self.client_selections),
                'avg_participation_rate': np.mean(list(participation_rates.values())),
            }
        
        # Statistical summary
        if len(self.train_accuracies) >= 10:
            early_acc = np.mean(self.train_accuracies[:5])
            late_acc = np.mean(self.train_accuracies[-5:])
            accuracy_improvement = late_acc - early_acc
        else:
            accuracy_improvement = 0
            
        report = {
            'experiment_info': {
                'dataset': args.dataset,
                'model': args.model,
                'total_rounds': args.epochs,
                'num_clients': args.num_users,
                'client_fraction': args.frac,
                'iid': args.iid,
                'alpha': getattr(args, 'alpha', 'NA'),
                'learning_rate': args.lr,
                'local_epochs': args.local_ep,
                'batch_size': args.local_bs,
                'optimizer': getattr(args, 'optimizer', 'SGD'),
                'experiment_seed': experiment_seed,
                'total_time_seconds': total_time
            },
            'performance_metrics': {
                **convergence_metrics,
                'test_accuracy': test_acc,
                'accuracy_improvement': accuracy_improvement,
                'avg_round_time': np.mean(self.round_times) if hasattr(self, 'round_times') and self.round_times else 0
            },
            'client_selection_analysis': client_analysis
        }
        
        return report

    def generate_comprehensive_report(self, args, test_acc, total_time, experiment_seed=None):
        """Generate a comprehensive analysis report."""
        convergence_metrics = self.calculate_convergence_metrics()
        client_analysis = self.analyze_client_selection_quality()
        memory_analysis = self.analyze_memory_bank_effectiveness()
        
        # Statistical summary
        if len(self.train_accuracies) >= 10:
            early_acc = np.mean(self.train_accuracies[:5])
            late_acc = np.mean(self.train_accuracies[-5:])
            accuracy_improvement = late_acc - early_acc
        else:
            accuracy_improvement = 0
            
        report = {
            'experiment_info': {
                'dataset': args.dataset,
                'model': args.model,
                'total_rounds': args.epochs,
                'num_clients': args.num_users,
                'client_fraction': args.frac,
                'iid': args.iid,
                'alpha': getattr(args, 'alpha', 'NA'),
                'learning_rate': args.lr,
                'local_epochs': args.local_ep,
                'batch_size': args.local_bs,
                # 🆕 USE ACTUAL VALUES TRACKED DURING EXECUTION
                'pumb_exploration_ratio': self.actual_config.get('pumb_exploration_ratio', getattr(args, 'pumb_exploration_ratio', 'NA')),
                'pumb_initial_rounds': self.actual_config.get('pumb_initial_rounds', getattr(args, 'pumb_initial_rounds', 'NA')),
                'quality_metric_type': self.actual_config.get('quality_metric_type', 'Unknown'),
                'embedding_dim': self.actual_config.get('embedding_dim', 'NA'),
                'experiment_seed': experiment_seed,
                'total_time_seconds': total_time
            },
            'performance_metrics': {
                **convergence_metrics,
                'test_accuracy': test_acc,
                'accuracy_improvement': accuracy_improvement,
                'avg_round_time': np.mean(self.round_times) if self.round_times else 0
            },
            'client_selection_analysis': client_analysis,
            'memory_bank_analysis': memory_analysis,
            'aggregation_weights_stats': {
                'num_rounds_tracked': len(self.aggregation_weights_history),
                'avg_weight_std': np.mean([np.std(list(weights.values())) 
                                         for weights in self.aggregation_weights_history]) 
                                         if self.aggregation_weights_history else 0
            },
            'quality_metric_config': {
            'alpha_loss_weight': self.actual_config.get('quality_alpha', getattr(args, 'quality_alpha', 'NA')),
            'beta_consistency_weight': self.actual_config.get('quality_beta', getattr(args, 'quality_beta', 'NA')), 
            'gamma_data_weight': self.actual_config.get('quality_gamma', getattr(args, 'quality_gamma', 'NA')),
            'metric_type': self.actual_config.get('quality_metric_type', 'Unknown')
            }
        }
        
        return report


def write_comprehensive_analysis(analyzer, args, test_acc, total_time, filename, experiment_seed=None):
    """Write comprehensive analysis to file."""
    report = analyzer.generate_comprehensive_report(args, test_acc, total_time, experiment_seed)
    
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE PUMB FEDERATED LEARNING ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Experiment Configuration
        f.write("EXPERIMENT CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        for key, value in report['experiment_info'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        # 🆕 ADD QUALITY METRIC PARAMETERS SECTION
        f.write("QUALITY METRIC PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        quality_config = report.get('quality_metric_config', {})
        f.write(f"Alpha (Loss Weight): {quality_config.get('alpha_loss_weight', 'NA')}\n")
        f.write(f"Beta (Consistency Weight): {quality_config.get('beta_consistency_weight', 'NA')}\n")
        f.write(f"Gamma (Data Weight): {quality_config.get('gamma_data_weight', 'NA')}\n")
        # ADD: Baseline quality parameter
        if hasattr(args, 'quality_baseline'):
            f.write(f"Baseline Quality: {args.quality_baseline}\n")
        elif hasattr(args, 'baseline_quality'):
            f.write(f"Baseline Quality: {args.baseline_quality}\n")
        f.write(f"Quality Metric Type: {quality_config.get('metric_type', 'Unknown')}\n")
        f.write("Note: α controls loss-based quality, β controls consistency, γ controls data quantity influence.\n")
        f.write("\n")
        
        # 1. Performance Comparison (Core Evidence)
        f.write("1. PERFORMANCE COMPARISON (CORE EVIDENCE):\n")
        f.write("-" * 50 + "\n")
        perf = report['performance_metrics']
        f.write(f"Final Test Accuracy: {perf['test_accuracy']:.4f} ({perf['test_accuracy']*100:.2f}%)\n")
        
        # FIX: Safe access to convergence metrics
        if 'convergence_round' in perf and 'target_ratio' in perf:
            f.write(f"Convergence Speed: {perf['convergence_round']} rounds to reach {perf['target_ratio']*100:.0f}% of final accuracy\n")
        else:
            f.write("Convergence Speed: Not calculated (insufficient convergence data)\n")
        
        f.write(f"Training Stability: {perf.get('training_stability', 0.0):.6f} (std of last 10 rounds)\n")
        f.write(f"Total Accuracy Improvement: {perf.get('accuracy_improvement', 0.0):.4f}\n")
        f.write(f"Average Round Time: {perf.get('avg_round_time', 0.0):.2f} seconds\n")
        f.write("\n")
        
        # 2. Client Selection Quality Analysis
        f.write("2. CLIENT SELECTION QUALITY ANALYSIS:\n")
        f.write("-" * 50 + "\n")
        client = report.get('client_selection_analysis', {})
        
        if client:
            f.write(f"Total Unique Clients Selected: {client.get('total_unique_clients', 0)}\n")
            f.write(f"Average Participation Rate: {client.get('avg_participation_rate', 0.0):.4f}\n")
            f.write(f"Top Quartile Participation: {client.get('top_quartile_participation', 0.0):.4f}\n")
            f.write(f"Bottom Quartile Participation: {client.get('bottom_quartile_participation', 0.0):.4f}\n")
            f.write(f"Participation Inequality: {client.get('participation_inequality', 0.0):.4f}\n")
            f.write(f"Average Reliability Improvement: {client.get('avg_reliability_improvement', 0.0):.4f}\n")
            f.write(f"Clients with Reliability Data: {client.get('clients_with_reliability_data', 0)}\n")
            f.write("\nClient Quality Score Statistics:\n")
            quality = client.get('quality_score_stats', {})
            f.write(f"  Mean Quality: {quality.get('mean', 0.0):.4f}\n")
            f.write(f"  Std Quality: {quality.get('std', 0.0):.4f}\n")
            f.write(f"  Min Quality: {quality.get('min', 0.0):.4f}\n")
            f.write(f"  Max Quality: {quality.get('max', 0.0):.4f}\n")
            
            # 🆕 ADD QUALITY METRIC INTERPRETATION
            f.write("\nQuality Metric Interpretation:\n")
            alpha = quality_config.get('alpha_loss_weight', 0.3)
            beta = quality_config.get('beta_consistency_weight', 0.2)
            gamma = quality_config.get('gamma_data_weight', 0.5)
            if isinstance(alpha, (int, float)) and isinstance(beta, (int, float)) and isinstance(gamma, (int, float)):
                f.write(f"  Loss-based selection influence: {alpha*100:.0f}%\n")
                f.write(f"  Consistency-based influence: {beta*100:.0f}%\n")
                f.write(f"  Data quantity influence: {gamma*100:.0f}%\n")
                
        else:
            f.write("Client Selection Analysis: No data available\n")
        f.write("\n")
        
        # 3. Memory Bank Effectiveness
        f.write("3. MEMORY BANK EFFECTIVENESS:\n")
        f.write("-" * 50 + "\n")
        memory = report.get('memory_bank_analysis', {})
        
        # Check if memory bank data exists (PUMB only)
        if memory and 'initial_memory_size' in memory:
            f.write(f"Initial Memory Size: {memory['initial_memory_size']}\n")
            f.write(f"Final Memory Size: {memory['final_memory_size']}\n")
            f.write(f"Memory Growth Rate: {memory['memory_growth_rate']:.2f} entries/round\n")
            f.write(f"Average Similarity Score: {memory['avg_similarity_score']:.4f}\n")
            f.write(f"Similarity Trend: {memory['similarity_trend']:.6f} (slope)\n")
            f.write(f"Ranking Stability: {memory['ranking_stability']:.4f} (correlation)\n")
            f.write(f"Rounds Tracked: {memory['total_rounds_tracked']}\n")
            
            # 🆕 ADD MEMORY BANK INTERPRETATION
            f.write("\nMemory Bank Interpretation:\n")
            if memory['avg_similarity_score'] > 0.7:
                f.write("  High similarity scores indicate effective parameter update learning\n")
            elif memory['avg_similarity_score'] > 0.5:
                f.write("  Moderate similarity scores suggest reasonable parameter learning\n")
            else:
                f.write("  Low similarity scores may indicate diverse or noisy parameter updates\n")
                
            if memory['similarity_trend'] > 0:
                f.write("  Positive similarity trend shows improving parameter update quality\n")
            elif memory['similarity_trend'] < -0.001:
                f.write("  Negative similarity trend suggests declining parameter coherence\n")
            else:
                f.write("  Stable similarity trend indicates consistent parameter update patterns\n")
        else:
            f.write("Memory Bank: Not applicable (FedAvg baseline) or insufficient data\n")
            f.write("This analysis is specific to PUMB algorithm.\n")
        f.write("\n")
        
        # 4. Aggregation Weights Analysis
        f.write("4. AGGREGATION WEIGHTS ANALYSIS:\n")
        f.write("-" * 50 + "\n")
        agg = report.get('aggregation_weights_stats', {})
        f.write(f"Rounds with Weight Data: {agg.get('num_rounds_tracked', 0)}\n")
        f.write(f"Average Weight Standard Deviation: {agg.get('avg_weight_std', 0.0):.6f}\n")
        
        # 🆕 ADD WEIGHT ANALYSIS INTERPRETATION
        if agg.get('avg_weight_std', 0) > 0:
            if agg['avg_weight_std'] > 0.1:
                f.write("  High weight variance indicates strong client quality differentiation\n")
            elif agg['avg_weight_std'] > 0.05:
                f.write("  Moderate weight variance shows selective client weighting\n")
            else:
                f.write("  Low weight variance suggests near-uniform client treatment\n")
        f.write("\n")
        
        # 🆕 ADD PUMB ALGORITHM EFFECTIVENESS SUMMARY
        f.write("5. PUMB ALGORITHM EFFECTIVENESS SUMMARY:\n")
        f.write("-" * 50 + "\n")
        
        # Summarize key insights
        has_memory_data = memory and 'initial_memory_size' in memory
        has_quality_data = client and client.get('quality_score_stats', {}).get('mean', 0) > 0
        
        if has_memory_data and has_quality_data:
            f.write("PUMB Features Successfully Activated:\n")
            f.write("✓ Memory bank tracking parameter update similarities\n")
            f.write("✓ Quality-based client selection with α/β/γ weighting\n")
            f.write("✓ Intelligent aggregation based on client reliability\n")
            
            # Performance assessment
            final_acc = perf['test_accuracy']
            if final_acc > 0.5:  # Assuming normalized accuracy
                f.write(f"✓ Strong performance: {final_acc*100:.2f}% test accuracy\n")
            elif final_acc > 0.3:
                f.write(f"○ Moderate performance: {final_acc*100:.2f}% test accuracy\n")
            else:
                f.write(f"△ Performance needs improvement: {final_acc*100:.2f}% test accuracy\n")
                
        elif has_quality_data:
            f.write("Partial PUMB Activation:\n")
            f.write("✓ Quality-based client selection active\n")
            f.write("△ Memory bank data limited or unavailable\n")
        else:
            f.write("Limited PUMB Data:\n")
            f.write("△ Quality metrics and memory bank data unavailable\n")
            f.write("△ Possible fallback to baseline federated averaging\n")
        f.write("\n")
        
        # Raw data summary
        f.write("6. RAW DATA SUMMARY:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Training Accuracy History: {len(analyzer.train_accuracies)} points\n")
        f.write(f"Test Accuracy Measurements: {len(analyzer.test_accuracies)} points\n")
        f.write(f"Memory Bank Size History: {len(analyzer.memory_bank_sizes)} points\n")
        f.write(f"Client Selection History: {len(analyzer.client_selections)} unique clients\n")
        f.write(f"Reliability History: {len(analyzer.client_reliability_history)} clients tracked\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")


def write_fedavg_comprehensive_analysis(analyzer, args, test_acc, total_time, filename, experiment_seed=None):
    """Write comprehensive analysis to file for FedAvg (without PUMB-specific features)."""
    report = analyzer.generate_fedavg_report(args, test_acc, total_time, experiment_seed)
    
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE FEDAVG FEDERATED LEARNING ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Experiment Configuration
        f.write("EXPERIMENT CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        for key, value in report['experiment_info'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        # 1. Performance Comparison (Core Evidence)
        f.write("1. PERFORMANCE COMPARISON (CORE EVIDENCE):\n")
        f.write("-" * 50 + "\n")
        perf = report['performance_metrics']
        f.write(f"Final Test Accuracy: {perf['test_accuracy']:.4f} ({perf['test_accuracy']*100:.2f}%)\n")
        
        if 'convergence_round' in perf and 'target_ratio' in perf:
            f.write(f"Convergence Speed: {perf['convergence_round']} rounds to reach {perf['target_ratio']*100:.0f}% of final accuracy\n")
        else:
            f.write("Convergence Speed: Not calculated (insufficient convergence data)\n")
        
        f.write(f"Training Stability: {perf.get('training_stability', 0.0):.6f} (std of last 10 rounds)\n")
        f.write(f"Total Accuracy Improvement: {perf.get('accuracy_improvement', 0.0):.4f}\n")
        f.write(f"Average Round Time: {perf.get('avg_round_time', 0.0):.2f} seconds\n")
        f.write("\n")
        
        # 2. Client Selection Analysis (FedAvg uses random selection)
        f.write("2. CLIENT SELECTION ANALYSIS:\n")
        f.write("-" * 50 + "\n")
        client = report.get('client_selection_analysis', {})
        
        if client:
            f.write(f"Total Unique Clients Selected: {client.get('total_unique_clients', 0)}\n")
            f.write(f"Average Participation Rate: {client.get('avg_participation_rate', 0.0):.4f}\n")
            f.write(f"Selection Method: Random (uniform probability)\n")
            f.write(f"Client Fraction per Round: {args.frac}\n")
        else:
            f.write("Client Selection Analysis: No data available\n")
        f.write("\n")
        
        # 3. Aggregation Method (FedAvg specific)
        f.write("3. AGGREGATION METHOD:\n")
        f.write("-" * 50 + "\n")
        f.write("Aggregation Strategy: Weighted Average (FedAvg)\n")
        f.write("Weight Calculation: Proportional to client data size\n")
        f.write("Memory Bank: Not applicable (FedAvg baseline)\n")
        f.write("Similarity Computation: Not applicable (FedAvg baseline)\n")
        f.write("Reliability Tracking: Not applicable (FedAvg baseline)\n")
        f.write("\n")
        
        # 4. Training Dynamics
        f.write("4. TRAINING DYNAMICS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Local Updates per Round: {args.local_ep} epochs\n")
        f.write(f"Local Batch Size: {args.local_bs}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Optimizer: {getattr(args, 'optimizer', 'SGD')}\n")
        if hasattr(analyzer, 'round_times') and analyzer.round_times:
            f.write(f"Average Round Time: {np.mean(analyzer.round_times):.2f} seconds\n")
        f.write("\n")
        
        # 5. Raw Data Summary
        f.write("5. RAW DATA SUMMARY:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Training Accuracy History: {len(analyzer.train_accuracies)} points\n")
        f.write(f"Training Loss History: {len(getattr(analyzer, 'train_losses', []))} points\n")
        f.write(f"Test Accuracy Measurements: {len(getattr(analyzer, 'test_accuracies', []))} points\n")
        f.write(f"Client Selection History: {len(analyzer.client_selections)} unique clients\n")
        f.write(f"Total Communication Rounds: {args.epochs}\n")
        f.write("\n")
        
        # 6. Statistical Significance Notes
        f.write("6. STATISTICAL SIGNIFICANCE NOTES:\n")
        f.write("-" * 50 + "\n")
        f.write("For statistical significance, run this experiment 3-5 times with different seeds.\n")
        f.write("Collect mean ± std for final accuracy and perform t-tests between methods.\n")
        if experiment_seed:
            f.write(f"Current experiment seed: {experiment_seed}\n")
        f.write("Compare against PUMB and other advanced federated learning methods.\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")

        
def aggregate_multiple_runs(reports_list, output_filename):
    """Aggregate results from multiple experimental runs for statistical significance."""
    if not reports_list:
        return
        
    # Extract key metrics from all runs
    final_accuracies = [r['performance_metrics']['test_accuracy'] for r in reports_list]
    convergence_rounds = [r['performance_metrics']['convergence_round'] for r in reports_list]
    training_stabilities = [r['performance_metrics']['training_stability'] for r in reports_list]
    
    # Calculate statistics
    acc_mean, acc_std = np.mean(final_accuracies), np.std(final_accuracies)
    conv_mean, conv_std = np.mean(convergence_rounds), np.std(convergence_rounds)
    stab_mean, stab_std = np.mean(training_stabilities), np.std(training_stabilities)
    
    # Confidence intervals (95%)
    n = len(final_accuracies)
    acc_ci = stats.t.interval(0.95, n-1, loc=acc_mean, scale=acc_std/np.sqrt(n))
    
    with open(output_filename, 'w') as f:
        f.write("STATISTICAL SIGNIFICANCE ANALYSIS - MULTIPLE RUNS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Number of experimental runs: {n}\n\n")
        
        f.write("FINAL TEST ACCURACY:\n")
        f.write(f"Mean ± Std: {acc_mean:.4f} ± {acc_std:.4f}\n")
        f.write(f"95% Confidence Interval: [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]\n")
        f.write(f"Individual runs: {[f'{acc:.4f}' for acc in final_accuracies]}\n\n")
        
        f.write("CONVERGENCE SPEED:\n")
        f.write(f"Mean ± Std: {conv_mean:.1f} ± {conv_std:.1f} rounds\n")
        f.write(f"Individual runs: {convergence_rounds}\n\n")
        
        f.write("TRAINING STABILITY:\n")
        f.write(f"Mean ± Std: {stab_mean:.6f} ± {stab_std:.6f}\n")
        f.write(f"Individual runs: {[f'{stab:.6f}' for stab in training_stabilities]}\n\n")

def write_scaffold_comprehensive_analysis(analyzer, args, final_test_acc, total_time, filename, experiment_seed=None):
    """
    Generate comprehensive analysis report for SCAFFOLD federated learning
    """
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SCAFFOLD FEDERATED LEARNING - COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment Seed: {experiment_seed}\n\n")
        
        # Experiment Configuration
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Algorithm: SCAFFOLD (Stochastic Controlled Averaging)\n")
        f.write(f"Dataset: {args.dataset.upper()}\n")
        f.write(f"Model: {args.model.upper()}\n")
        f.write(f"Global Rounds: {args.epochs}\n")
        f.write(f"Total Clients: {args.num_users}\n")
        f.write(f"Client Participation Rate: {args.frac} ({int(args.frac * args.num_users)} clients/round)\n")
        f.write(f"Local Epochs: {args.local_ep}\n")
        f.write(f"Local Batch Size: {args.local_bs}\n")
        f.write(f"Learning Rate: {getattr(args, 'lr', 'N/A')}\n")
        f.write(f"Optimizer: {getattr(args, 'optimizer', 'N/A')}\n")
        f.write(f"Data Distribution: {'IID' if args.iid else 'Non-IID'}\n")
        if not args.iid:
            f.write(f"Dirichlet Alpha: {getattr(args, 'alpha', 'N/A')}\n")
        f.write(f"SCAFFOLD Step Size: {getattr(args, 'scaffold_stepsize', 1.0)}\n")
        f.write("\n")
        
        # Performance Metrics
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Final Test Accuracy: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)\n")
        f.write(f"Total Training Time: {total_time:.2f} seconds\n")
        f.write(f"Average Time per Round: {total_time/args.epochs:.2f} seconds\n")
        f.write("\n")
        
        # Convergence Analysis
        convergence_metrics = analyzer.calculate_convergence_metrics()
        f.write("CONVERGENCE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Convergence Round: {convergence_metrics.get('convergence_round', 'N/A')}\n")
        f.write(f"Training Stability: {convergence_metrics.get('training_stability', 0):.6f}\n")
        f.write(f"Final Training Accuracy: {convergence_metrics.get('final_train_acc', 0)*100:.2f}%\n")
        f.write(f"Training Loss Reduction: {convergence_metrics.get('loss_reduction', 0):.4f}\n")
        f.write("\n")
        
        # Client Selection Analysis
        client_analysis = analyzer.analyze_client_selection_quality()
        f.write("CLIENT SELECTION ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Unique Clients Selected: {client_analysis.get('total_unique_clients', 0)}\n")
        f.write(f"Average Participation Rate: {client_analysis.get('avg_participation_rate', 0):.4f}\n")
        f.write(f"Client Selection Diversity: {client_analysis.get('selection_diversity', 0):.4f}\n")
        f.write("Note: SCAFFOLD uses random client selection with control variates for gradient correction\n")
        f.write("\n")
        
        # Control Variate Analysis (SCAFFOLD-specific)
        f.write("SCAFFOLD-SPECIFIC ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("Control Variate Behavior:\n")
        f.write("- SCAFFOLD uses control variates to reduce client drift in non-IID settings\n")
        f.write("- Global control variates are updated based on aggregated local corrections\n")
        f.write("- Local control variates help clients correct for distribution heterogeneity\n")
        f.write(f"- Random client selection ensures unbiased gradient estimation\n")
        f.write("\n")
        
        # Algorithm Strengths and Limitations
        f.write("ALGORITHM CHARACTERISTICS\n")
        f.write("-" * 40 + "\n")
        f.write("SCAFFOLD Strengths:\n")
        f.write("+ Reduces client drift through control variates\n")
        f.write("+ Theoretically proven convergence guarantees\n")
        f.write("+ Works with arbitrary local update steps\n")
        f.write("+ Handles non-IID data distributions\n")
        f.write("\n")
        f.write("SCAFFOLD Limitations:\n")
        f.write("- Requires additional memory for control variates\n")
        f.write("- Sensitive to learning rate and hyperparameter tuning\n")
        f.write("- Can be unstable with aggressive learning rates\n")
        f.write("- Random client selection may miss important clients\n")
        f.write("\n")
        
        # Round-by-Round Summary (last 10 rounds)
        f.write("FINAL ROUNDS SUMMARY (Last 10 Rounds)\n")
        f.write("-" * 40 + "\n")
        if hasattr(analyzer, 'round_data') and len(analyzer.round_data) >= 10:
            for i in range(max(0, len(analyzer.round_data)-10), len(analyzer.round_data)):
                round_data = analyzer.round_data[i]
                f.write(f"Round {round_data['round_num']:3d}: ")
                f.write(f"Train Acc={round_data.get('train_acc', 0)*100:5.2f}% ")
                if round_data.get('test_acc') is not None:
                    f.write(f"Test Acc={round_data['test_acc']*100:5.2f}% ")
                f.write(f"Loss={round_data.get('train_loss', 0):6.4f} ")
                f.write(f"Time={round_data.get('round_time', 0):4.1f}s")
                f.write(f" Clients={len(round_data.get('selected_clients', []))}\n")
        f.write("\n")
        
        # Research Insights
        f.write("RESEARCH INSIGHTS FOR COMPARISON\n")
        f.write("-" * 40 + "\n")
        f.write("SCAFFOLD Performance Characteristics:\n")
        f.write(f"- Achieved {final_test_acc*100:.2f}% accuracy with control variate correction\n")
        f.write(f"- Uses gradient correction to handle client heterogeneity\n")
        f.write(f"- Random client selection provides unbiased but potentially suboptimal coverage\n")
        f.write(f"- Performance depends heavily on hyperparameter tuning and stability\n")
        f.write("\n")
        f.write("For comparison with intelligent client selection methods (e.g., PUMB):\n")
        f.write("- SCAFFOLD focuses on gradient correction rather than client quality\n")
        f.write("- May benefit from combining with intelligent client selection\n")
        f.write("- Control variate overhead vs. client selection efficiency trade-off\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("END OF SCAFFOLD COMPREHENSIVE ANALYSIS\n")
        f.write("="*80 + "\n")

