import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import skew, kurtosis

class EmbeddingGenerator:
    def __init__(self, feature_dim=13, embedding_dim=512):
        """Initialize the parameter update embedding generator."""
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        # Network architecture
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, embedding_dim)
        )
        
        # Weight initialization
        with torch.no_grad():
            for layer in self.projection:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    torch.nn.init.zeros_(layer.bias)
    
    def generate_embedding(self, parameter_update):
        """Generate embedding from parameter update"""
        # Extract features
        features = self._extract_features(parameter_update)
        
        # Convert to tensor
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        
        # Handle NaN or infinite values
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Scale features
        features = self._scale_features(features)
        
        # Project to embedding space
        self.projection.eval()
        with torch.no_grad():
            # Ensure batch dimension
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            
            embedding = self.projection(features).squeeze(0)
        
        # Add controlled noise for discrimination
        noise_std = 0.02
        noise = torch.normal(0, noise_std, size=embedding.shape)
        embedding = embedding + noise
        
        # Normalize for cosine similarity
        embedding = F.normalize(embedding, p=2, dim=0)
        
        return embedding

    def _scale_features(self, features):
        """Scale features using robust scaling"""
        # Simple robust scaling
        median_val = torch.median(features)
        mad = torch.median(torch.abs(features - median_val))
        
        if mad > 1e-8:
            scaled = (features - median_val) / (mad * 1.4826 + 1e-8)
        else:
            scaled = features - median_val
        
        # Gentle non-linear transformation
        scaled = torch.tanh(scaled * 0.3)
        
        return scaled

    def _extract_features(self, parameter_update):
        """Extract features from parameter update"""
        # Single conversion to numpy array
        if isinstance(parameter_update, torch.Tensor):
            update_values = parameter_update.detach().cpu().numpy().flatten()
        elif isinstance(parameter_update, dict):
            # Efficient concatenation for state dicts
            arrays = []
            for v in parameter_update.values():
                if isinstance(v, torch.Tensor):
                    arrays.append(v.detach().cpu().numpy().flatten())
                else:
                    arrays.append(np.array(v).flatten())
            update_values = np.concatenate(arrays) if arrays else np.array([])
        else:
            update_values = np.array(parameter_update).flatten()
        
        # Handle edge cases
        if len(update_values) == 0:
            return np.random.normal(0, 0.1, size=self.feature_dim)
        
        # Remove non-finite values
        finite_mask = np.isfinite(update_values)
        if not finite_mask.any():
            return np.random.normal(0, 0.1, size=self.feature_dim)
        
        update_values = update_values[finite_mask]
        
        if len(update_values) == 0:
            return np.random.normal(0, 0.1, size=self.feature_dim)
        
        # Vectorized feature computation
        try:
            # Pre-compute commonly used values
            abs_values = np.abs(update_values)
            squared_values = update_values ** 2
            
            # Compute percentiles in one call
            percentiles = np.percentile(update_values, [25, 50, 75])
            q25, median_val, q75 = percentiles
            
            # All features computed efficiently
            features = np.array([
                np.mean(update_values),                    # mean
                np.std(update_values),                     # std
                np.linalg.norm(update_values, 2),          # l2 norm
                np.linalg.norm(update_values, 1),          # l1 norm
                np.min(update_values),                     # min
                np.max(update_values),                     # max
                median_val,                                # median
                skew(update_values),                       # skewness
                q75 - q25,                                 # IQR
                np.mean(abs_values),                       # mean absolute value
                kurtosis(update_values),                   # kurtosis
                np.sum(abs_values < 1e-6) / len(update_values),  # sparsity
                np.sum(squared_values)                     # energy
            ])
            
            # Handle any remaining NaN values
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        except Exception as e:
            # Fallback to random features
            features = np.random.normal(0, 0.01, size=self.feature_dim)
        
        return features[:self.feature_dim]