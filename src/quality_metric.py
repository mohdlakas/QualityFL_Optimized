import numpy as np
import torch
import logging

class QualityMetric:
    """Quality metric for client assessment"""

    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1, baseline_quality=0.4):
        self.alpha = alpha  # Loss improvement weight
        self.beta = beta    # Consistency weight  
        self.gamma = gamma  # Data size weight
        self.baseline_quality = baseline_quality
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1"
        self.logger = logging.getLogger('PUMB_Quality')
        
        # Track statistics for better normalization
        self.loss_improvements_history = []
        self.consistency_scores_history = []

    def calculate_quality(self, loss_before, loss_after, data_sizes, param_update, 
                         round_num, client_id, all_loss_improvements=None):
        """Calculate quality score for a client"""
        # Calculate loss improvement
        loss_improvement = max(0, loss_before - loss_after)
        
        # Better normalization using all clients in current round
        if all_loss_improvements is not None and len(all_loss_improvements) > 0:
            if len(all_loss_improvements) > 1:
                improvement_75th = np.percentile(all_loss_improvements, 75)
                Q_loss = min(1.0, loss_improvement / (improvement_75th + 1e-8))
            else:
                Q_loss = 1.0 if loss_improvement > 0 else 0.1
        else:
            # For standalone calculation, use relative improvement
            relative_improvement = loss_improvement / (loss_before + 1e-8)
            Q_loss = min(1.0, relative_improvement * 10)
        
        # Robust consistency calculation
        Q_consistency = self._calculate_consistency(param_update)
        
        # Data quality calculation
        Q_data = self._calculate_data_quality(data_sizes, client_id)
        
        # Combine with weighting
        quality = self.alpha * Q_loss + self.beta * Q_consistency + self.gamma * Q_data
        quality = max(0.1, min(1.0, quality))
        
        # Track for normalization
        self.consistency_scores_history.append(Q_consistency)
        if len(self.consistency_scores_history) > 100:
            self.consistency_scores_history.pop(0)
            
        return quality
    
    def calculate_quality_batch(self, client_data_batch, round_num):
        """Batch process quality scores for efficiency"""
        qualities = {}
        
        # Extract all loss improvements for batch normalization
        all_improvements = []
        all_data_sizes = []
        
        for client_id, data in client_data_batch.items():
            loss_improvement = max(0, data['loss_before'] - data['loss_after'])
            all_improvements.append(loss_improvement)
            all_data_sizes.append(data.get('data_size', 100))
        
        # Compute percentiles ONCE for all clients
        if len(all_improvements) > 1:
            improvement_75th = np.percentile(all_improvements, 75)
            data_75th = np.percentile(all_data_sizes, 75)
        else:
            improvement_75th = max(all_improvements) if all_improvements else 0.001
            data_75th = max(all_data_sizes) if all_data_sizes else 100
        
        # Process all clients with shared statistics
        for client_id, data in client_data_batch.items():
            loss_improvement = max(0, data['loss_before'] - data['loss_after'])
            
            # Q_loss with batch normalization
            Q_loss = min(1.0, loss_improvement / (improvement_75th + 1e-8))
            
            # Consistency calculation
            Q_consistency = self._calculate_consistency(data['param_update'])
            
            # Data quality
            client_data_size = data.get('data_size', 100)
            Q_data = min(1.0, client_data_size / (data_75th + 1e-8))
            
            # Combine with proper weighting
            quality = self.alpha * Q_loss + self.beta * Q_consistency + self.gamma * Q_data
            qualities[client_id] = max(0.1, min(1.0, quality))
        
        return qualities
    
    def _calculate_consistency(self, param_update):
        """Calculate consistency score from parameter updates"""
        try:
            # Extract parameter values
            if isinstance(param_update, dict):
                param_values = torch.cat([p.flatten() for p in param_update.values()])
            else:
                param_values = param_update.flatten()
                
            param_np = param_values.detach().cpu().numpy()
            
            if len(param_np) == 0:
                return 0.5
            
            # Remove outliers
            param_np = param_np[np.abs(param_np) < np.percentile(np.abs(param_np), 95)]
            
            if len(param_np) < 10:
                return 0.5
                
            mean_val = np.mean(param_np)
            std_val = np.std(param_np)
            
            # Multiple consistency measures
            # 1. Coefficient of variation
            if abs(mean_val) > 1e-8:
                cv = std_val / abs(mean_val)
                consistency_1 = np.exp(-cv)
            else:
                consistency_1 = 0.5
                
            # 2. Normalized standard deviation
            param_range = np.max(param_np) - np.min(param_np)
            if param_range > 1e-8:
                normalized_std = std_val / param_range
                consistency_2 = 1.0 - min(1.0, normalized_std)
            else:
                consistency_2 = 1.0
                
            # 3. Sparsity-based consistency
            sparsity = np.sum(np.abs(param_np) < 1e-6) / len(param_np)
            sparsity_penalty = 1.0 - min(0.5, sparsity)
            
            # Combine measures
            Q_consistency = (consistency_1 + consistency_2 + sparsity_penalty) / 3.0
            
        except Exception as e:
            self.logger.warning(f"Consistency calculation failed: {e}, using default")
            Q_consistency = 0.5
            
        return max(0.1, min(1.0, Q_consistency))
    
    def _calculate_data_quality(self, data_sizes, client_id):
        """Calculate data quality based on data size"""
        if not data_sizes or client_id not in data_sizes:
            return 0.5
            
        client_data_size = data_sizes[client_id]
        all_sizes = list(data_sizes.values())
        
        if len(all_sizes) <= 1:
            return 1.0
            
        # Use relative positioning
        sorted_sizes = sorted(all_sizes)
        client_percentile = (sorted_sizes.index(client_data_size) + 1) / len(sorted_sizes)
        
        # Clients in top 50% get higher data quality scores
        if client_percentile >= 0.5:
            Q_data = 0.5 + 0.5 * ((client_percentile - 0.5) / 0.5)
        else:
            Q_data = 0.3 + 0.2 * (client_percentile / 0.5)
            
        return Q_data

class GenerousQualityMetric(QualityMetric):
    """Exploration-friendly quality metric with higher baseline scores"""
    
    def __init__(self, alpha=0.6, beta=0.2, gamma=0.2, baseline_quality=0.5): 
        super().__init__(alpha, beta, gamma, baseline_quality)
        self.baseline_quality = baseline_quality  # Ensure it's stored
            
        # Track quality statistics
        self.quality_history = []
        self.client_selections = {}
        
        print(f"🔍 GenerousQualityMetric initialized with baseline_quality: {self.baseline_quality}")
        
    def calculate_quality(self, loss_before, loss_after, data_sizes, param_update, 
                    round_num, client_id, all_loss_improvements=None):
        """
        FIXED: Proper baseline quality enforcement with clearer logic
        """
        # Calculate base quality using parent method
        base_quality = super().calculate_quality(
            loss_before, loss_after, data_sizes, param_update,
            round_num, client_id, all_loss_improvements
        )
        
        # Calculate bonuses
        exploration_bonus = 0.0
        if round_num < 10:
            exploration_bonus = 0.2 * (10 - round_num) / 10
            
        # Client diversity bonus
        if client_id not in self.client_selections:
            self.client_selections[client_id] = 0
        self.client_selections[client_id] += 1
        
        avg_selections = np.mean(list(self.client_selections.values())) if self.client_selections else 1
        client_usage = self.client_selections[client_id]
        diversity_bonus = max(0, 0.1 * (avg_selections - client_usage) / avg_selections)
        
        total_bonus = exploration_bonus + diversity_bonus
        
        # FIXED APPROACH: Apply baseline properly
        # Method 1: Baseline as minimum floor (most common interpretation)
        if base_quality < self.baseline_quality:
            # If base quality is below baseline, boost it to baseline + partial bonuses
            generous_quality = self.baseline_quality + (total_bonus * 0.5)
        else:
            # If base quality is above baseline, add full bonuses
            generous_quality = base_quality + total_bonus
        
        # Cap at 1.0
        generous_quality = min(1.0, generous_quality)
        
        # Enhanced debugging - print every few rounds or for specific clients
        if (round_num <= 5) or (round_num % 10 == 0 and client_id in [0, 1, 2]):
            print(f"🔍 R{round_num} C{client_id}: baseline={self.baseline_quality:.3f}, "
                  f"base={base_quality:.3f}, bonuses={total_bonus:.3f}, final={generous_quality:.3f}")
        
        self.quality_history.append(generous_quality)
        return generous_quality

    def calculate_quality_batch(self, client_data_batch, round_num):
        """Batch process with proper baseline enforcement"""
        qualities = {}
        
        # Get base qualities first
        base_qualities = super().calculate_quality_batch(client_data_batch, round_num)
        
        # Apply generous adjustments
        for client_id, base_quality in base_qualities.items():
            # Calculate bonuses (same logic as individual method)
            exploration_bonus = 0.0
            if round_num < 10:
                exploration_bonus = 0.2 * (10 - round_num) / 10
                
            if client_id not in self.client_selections:
                self.client_selections[client_id] = 0
            self.client_selections[client_id] += 1
            
            avg_selections = np.mean(list(self.client_selections.values())) if self.client_selections else 1
            client_usage = self.client_selections[client_id]
            diversity_bonus = max(0, 0.1 * (avg_selections - client_usage) / avg_selections)
            
            total_bonus = exploration_bonus + diversity_bonus
            
            # Apply same baseline logic
            if base_quality < self.baseline_quality:
                generous_quality = self.baseline_quality + (total_bonus * 0.5)
            else:
                generous_quality = base_quality + total_bonus
                
            qualities[client_id] = min(1.0, generous_quality)
        
        return qualities
