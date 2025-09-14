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
