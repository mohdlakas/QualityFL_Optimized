import numpy as np
import torch
import logging



class QualityMetric:
    """
    IMPROVED QUALITY METRIC with fixes for mathematical stability and better exploration
    
    PROBLEMS FIXED:
    1. Q_consistency formula is now mathematically stable
    2. More balanced weights (reduced loss bias)
    3. Proper handling of negative mean values in consistency calculation
    4. Robust division by small numbers with better fallbacks
    5. Cross-client normalization within each round
    """

    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1, baseline_quality=0.4):  # FIXED: More balanced weights
        """
        IMPROVED: More balanced quality assessment
        - Reduced loss weight from 0.6 to 0.4
        - Increased data weight from 0.1 to 0.3
        - This gives more value to clients with substantial data
        """
        self.alpha = alpha  # Loss improvement weight (reduced)
        self.beta = beta    # Consistency weight  
        self.gamma = gamma  # Data size weight (increased)
        self.baseline_quality = baseline_quality  # Store baseline quality for use in subclasses
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1"
        self.logger = logging.getLogger('PUMB_Quality')
        
        # ADDED: Track statistics for better normalization
        self.loss_improvements_history = []
        self.consistency_scores_history = []

    def calculate_quality(self, loss_before, loss_after, data_sizes, param_update, 
                         round_num, client_id, all_loss_improvements=None):
        """
        THEORY-ALIGNED with FIXES: Implement q_i^t = α·Q_loss + β·Q_consistency + γ·Q_data
        """
        # FIX 1: Better Q_loss calculation with relative improvement
        loss_improvement = max(0, loss_before - loss_after)  # ΔL_i^t
        
        # IMPROVED: Better normalization using all clients in current round
        if all_loss_improvements is not None and len(all_loss_improvements) > 0:
            # Use percentile-based normalization instead of max
            if len(all_loss_improvements) > 1:
                improvement_75th = np.percentile(all_loss_improvements, 75)
                Q_loss = min(1.0, loss_improvement / (improvement_75th + 1e-8))
            else:
                Q_loss = 1.0 if loss_improvement > 0 else 0.1
        else:
            # For standalone calculation, use relative improvement
            relative_improvement = loss_improvement / (loss_before + 1e-8)
            Q_loss = min(1.0, relative_improvement * 10)  # Scale up relative improvement
        
        # FIX 2: Robust Q_consistency calculation
        Q_consistency = self._calculate_robust_consistency(param_update)
        
        # FIX 3: Improved Q_data with relative scaling
        Q_data = self._calculate_data_quality(data_sizes, client_id)
        
        # FIX 4: Add quality floor to prevent extremely low scores
        quality = self.alpha * Q_loss + self.beta * Q_consistency + self.gamma * Q_data
        quality = max(0.1, min(1.0, quality))  # More generous floor (0.1 instead of 0.01)
        
        # Enhanced logging
        # self.logger.info(f"Client {client_id} R{round_num}: "
        #                 f"Q_loss={Q_loss:.4f}, Q_consistency={Q_consistency:.4f}, "
        #                 f"Q_data={Q_data:.4f}, final={quality:.4f}")
        
        # Track for normalization
        self.consistency_scores_history.append(Q_consistency)
        if len(self.consistency_scores_history) > 100:
            self.consistency_scores_history.pop(0)
            
        return quality
    
    def _calculate_robust_consistency(self, param_update):
        """
        FIX: Robust consistency calculation that handles edge cases
        """
        try:
            # Extract parameter values
            if isinstance(param_update, dict):
                param_values = torch.cat([p.flatten() for p in param_update.values()])
            else:
                param_values = param_update.flatten()
                
            param_np = param_values.detach().cpu().numpy()
            
            if len(param_np) == 0:
                return 0.5
            
            # Remove outliers for more stable calculation
            param_np = param_np[np.abs(param_np) < np.percentile(np.abs(param_np), 95)]
            
            if len(param_np) < 10:  # Too few values
                return 0.5
                
            mean_val = np.mean(param_np)
            std_val = np.std(param_np)
            
            # IMPROVED: Multiple consistency measures
            # 1. Coefficient of variation (robust to scale)
            if abs(mean_val) > 1e-8:
                cv = std_val / abs(mean_val)
                consistency_1 = np.exp(-cv)
            else:
                consistency_1 = 0.5  # Neutral for near-zero updates
                
            # 2. Normalized standard deviation
            param_range = np.max(param_np) - np.min(param_np)
            if param_range > 1e-8:
                normalized_std = std_val / param_range
                consistency_2 = 1.0 - min(1.0, normalized_std)
            else:
                consistency_2 = 1.0  # Perfect consistency for constant values
                
            # 3. Sparsity-based consistency (penalize too many zeros)
            sparsity = np.sum(np.abs(param_np) < 1e-6) / len(param_np)
            sparsity_penalty = 1.0 - min(0.5, sparsity)  # Don't penalize more than 50%
            
            # Combine measures
            Q_consistency = (consistency_1 + consistency_2 + sparsity_penalty) / 3.0
            
        except Exception as e:
            self.logger.warning(f"Consistency calculation failed: {e}, using default")
            Q_consistency = 0.5
            
        return max(0.1, min(1.0, Q_consistency))
    
    def _calculate_data_quality(self, data_sizes, client_id):
        """
        IMPROVED: More nuanced data size quality
        """
        if not data_sizes or client_id not in data_sizes:
            return 0.5
            
        client_data_size = data_sizes[client_id]
        all_sizes = list(data_sizes.values())
        
        if len(all_sizes) <= 1:
            return 1.0
            
        # Use relative positioning instead of just max normalization
        sorted_sizes = sorted(all_sizes)
        client_percentile = (sorted_sizes.index(client_data_size) + 1) / len(sorted_sizes)
        
        # Clients in top 50% get higher data quality scores
        if client_percentile >= 0.5:
            Q_data = 0.5 + 0.5 * ((client_percentile - 0.5) / 0.5)
        else:
            Q_data = 0.3 + 0.2 * (client_percentile / 0.5)
            
        return Q_data

    # Keep legacy methods for compatibility
    def _flatten_params(self, model_params):
        """Flatten PyTorch parameters into a single vector."""
        if isinstance(model_params, dict):
            # If it's a state dict
            return torch.cat([p.flatten() for p in model_params.values()])
        elif isinstance(model_params, list):
            # If it's a list of tensors
            return torch.cat([p.flatten() for p in model_params])
        else:
            # Assume it's already a tensor
            return model_params.flatten()
    
    def _flatten_params_numpy(self, model_params):
        """Flatten numpy parameters into a single vector."""
        if isinstance(model_params, dict):
            # If it's a dict
            return np.concatenate([p.flatten() for p in model_params.values()])
        elif isinstance(model_params, list):
            # If it's a list of arrays
            return np.concatenate([p.flatten() for p in model_params])
        else:
            # Assume it's already an array
            return model_params.flatten()

class CIFAR100QualityMetric(QualityMetric):
    """
    CIFAR-100 optimized quality metric with longer exploration phase
    """
    
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__(alpha, beta, gamma)
        self.cifar100_exploration_rounds = 15
        self.baseline_quality = 0.4
        self.round_qualities = []
        
        # Setup logging
        from datetime import datetime
        import os
        
        log_dir = '../save/logs'
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{log_dir}/cifar100_quality_{timestamp}.log'
        
        has_file_handler = any(isinstance(h, logging.FileHandler) for h in self.logger.handlers)
        
        if not has_file_handler:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)
            
            self.logger.info("=== CIFAR-100 QUALITY METRIC INITIALIZED ===")
            self.logger.info(f"Weights: alpha={alpha}, beta={beta}, gamma={gamma}")
            self.logger.info(f"Exploration rounds: {self.cifar100_exploration_rounds}")
            
        print(f"📝 CIFAR100QualityMetric logging to: {log_file}")
        
    def calculate_quality(self, loss_before, loss_after, data_sizes, param_update, 
                         round_num, client_id, all_loss_improvements=None):
        
        # Extended exploration phase for CIFAR-100
        if round_num < self.cifar100_exploration_rounds:
            exploration_quality = 0.35 + 0.3 * np.random.random()
            self.round_qualities.append(exploration_quality)
            return exploration_quality
            
        # Regular quality calculation with CIFAR-100 adjustments
        base_quality = super().calculate_quality(
            loss_before, loss_after, data_sizes, param_update,
            round_num, client_id, all_loss_improvements
        )
        
        # CIFAR-100 specific adjustments
        loss_improvement = max(0, loss_before - loss_after)
        if loss_improvement > 0.05:
            improvement_bonus = 0.15
        elif loss_improvement > 0.01:
            improvement_bonus = 0.05
        else:
            improvement_bonus = 0.0
            
        # Data size bonus
        if data_sizes and client_id in data_sizes:
            all_sizes = list(data_sizes.values())
            if len(all_sizes) > 1:
                client_size = data_sizes[client_id]
                size_percentile = (sorted(all_sizes).index(client_size) + 1) / len(all_sizes)
                if size_percentile > 0.7:
                    data_bonus = 0.1
                else:
                    data_bonus = 0.0
            else:
                data_bonus = 0.0
        else:
            data_bonus = 0.0
        
        adjusted_quality = max(self.baseline_quality, 
                              base_quality + improvement_bonus + data_bonus)
        
        self.round_qualities.append(adjusted_quality)
        
        if round_num % 10 == 0 and len(self.round_qualities) > 20:
            recent_q = self.round_qualities[-20:]
            self.logger.info(f"Round {round_num}: Recent quality mean={np.mean(recent_q):.4f}, "
                           f"std={np.std(recent_q):.4f}")
        
        return min(1.0, adjusted_quality)
    
    
# ALTERNATIVE: Even more generous quality metric for better exploration
class GenerousQualityMetric(QualityMetric):
    """
    EXPLORATION-FRIENDLY: Version that gives higher baseline scores
    to encourage more client participation
    """
    
    def __init__(self, alpha=0.6, beta=0.2, gamma=0.2, baseline_quality=0.6):  # CIFAR-100 optimized weights
        super().__init__(alpha, beta, gamma, baseline_quality)
        self.baseline_quality = baseline_quality  

        # # Set up file logging to same directory as comprehensive analysis
        # from datetime import datetime
        # import os
        
        # # Create logs directory if it doesn't exist
        # log_dir = '../save/logs'
        # os.makedirs(log_dir, exist_ok=True)
        
        # # Create timestamped log file in same location as comprehensive analysis
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # log_file = f'{log_dir}/generous_quality_{timestamp}.log'
        
        # # Check if file handler already exists to avoid duplicates
        # has_file_handler = any(isinstance(h, logging.FileHandler) for h in self.logger.handlers)
        
        # if not has_file_handler:
        #     file_handler = logging.FileHandler(log_file)
        #     file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        #     self.logger.addHandler(file_handler)
        #     self.logger.setLevel(logging.INFO)
            
        #     # Test log to confirm it's working
        #     self.logger.info("=== GENEROUS QUALITY METRIC INITIALIZED ===")
        #     self.logger.info(f"Weights: alpha={alpha}, beta={beta}, gamma={gamma}")
        #     self.logger.info(f"Baseline quality: {self.baseline_quality}")
        #     self.logger.info("ADDRESSING CIFAR-100 PERFORMANCE ISSUES")
            
        # print(f"📝 GenerousQualityMetric logging to: {log_file}")
        
        # Track quality statistics
        self.quality_history = []
        self.client_selections = {}
        
    def calculate_quality(self, loss_before, loss_after, data_sizes, param_update, 
                         round_num, client_id, all_loss_improvements=None):
        """
        ULTRA-GENEROUS: Much higher baseline quality to fix catastrophic performance
        """
        base_quality = super().calculate_quality(
            loss_before, loss_after, data_sizes, param_update,
            round_num, client_id, all_loss_improvements
        )
        
        # AGGRESSIVE exploration bonus for CIFAR-100
        exploration_bonus = 0.0
        if round_num < 10:  # Extended exploration phase
            exploration_bonus = 0.2 * (10 - round_num) / 10  # Larger bonus
            
        # Client diversity bonus - reward less frequently selected clients
        if client_id not in self.client_selections:
            self.client_selections[client_id] = 0
        self.client_selections[client_id] += 1
        
        # Bonus for underused clients
        avg_selections = np.mean(list(self.client_selections.values())) if self.client_selections else 1
        client_usage = self.client_selections[client_id]
        diversity_bonus = max(0, 0.1 * (avg_selections - client_usage) / avg_selections)
        
        # MUCH more generous quality calculation
        generous_quality = max(
            self.baseline_quality,  # 0.4 baseline
            base_quality + exploration_bonus + diversity_bonus
        )
        
        # Track statistics
        self.quality_history.append(generous_quality)
        
        # # Enhanced logging with performance context
        # self.logger.info(f"R{round_num} Client {client_id}: base={base_quality:.4f}, "
        #                 f"explore_bonus={exploration_bonus:.4f}, "
        #                 f"diversity_bonus={diversity_bonus:.4f}, "
        #                 f"final={generous_quality:.4f}, "
        #                 f"selections={client_usage}")
        
        # # Log statistics every 5 rounds for better monitoring
        # if round_num % 5 == 0 and len(self.quality_history) > 10:
        #     recent_qualities = self.quality_history[-50:]
        #     mean_q = np.mean(recent_qualities)
        #     std_q = np.std(recent_qualities)
        #     min_q = np.min(recent_qualities)
        #     max_q = np.max(recent_qualities)
            
        #     self.logger.info(f"=== QUALITY STATS ROUND {round_num} ===")
        #     self.logger.info(f"Recent qualities: mean={mean_q:.4f}, std={std_q:.4f}, "
        #                    f"min={min_q:.4f}, max={max_q:.4f}")
        #     self.logger.info(f"Client selections: {len(self.client_selections)} unique clients")
        #     self.logger.info(f"Selection distribution: min={min(self.client_selections.values())}, "
        #                    f"max={max(self.client_selections.values())}, "
        #                    f"avg={np.mean(list(self.client_selections.values())):.1f}")
            
            # # CRITICAL: Log if performance is still terrible
            # if round_num > 10 and mean_q < 0.5:
            #     self.logger.warning(f"LOW QUALITY SCORES DETECTED! Mean quality {mean_q:.4f} may cause poor performance")
                
        return min(1.0, generous_quality)


class StableQualityMetric(QualityMetric):
    """
    STABILITY-FOCUSED: Conservative quality metric that prioritizes stable training
    
    KEY FIXES:
    1. More balanced weights (less loss bias)
    2. Conservative baseline quality
    3. Gradual transition from exploration to exploitation
    4. Robust normalization to prevent extreme scores
    5. Connects to args for initial_rounds configuration
    """
    
    def __init__(self, alpha=0.5, beta=0.1, gamma=0.4, args=None):  # Accept args parameter
        super().__init__(alpha, beta, gamma)
        self.baseline_quality = 0.3  # Conservative baseline
        
        # Connect to args for initial rounds (default to 10 if args not provided)
        if args is not None:
            self.initial_rounds = getattr(args, 'pumb_initial_rounds', 15)
        else:
            self.initial_rounds = 15
        
        # Setup logging
        from datetime import datetime
        import os
        
        log_dir = '../save/logs'
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{log_dir}/stable_quality_{timestamp}.log'
        
        # Avoid duplicate handlers
        has_file_handler = any(isinstance(h, logging.FileHandler) for h in self.logger.handlers)
        
        if not has_file_handler:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)
            
            self.logger.info("=== STABLE QUALITY METRIC INITIALIZED ===")
            self.logger.info(f"Weights: alpha={alpha}, beta={beta}, gamma={gamma}")
            self.logger.info(f"Baseline quality: {self.baseline_quality}")
            self.logger.info(f"Initial rounds (exploration): {self.initial_rounds}")
            
        print(f"📝 StableQualityMetric logging to: {log_file}")
        print(f"🎯 Using {self.initial_rounds} initial rounds for exploration")
        
        # Track statistics for adaptive behavior
        self.round_qualities = []
        self.client_participation = {}
        self.global_loss_trend = []
        
    def calculate_quality(self, loss_before, loss_after, data_sizes, param_update, 
                         round_num, client_id, all_loss_improvements=None):
        """
        STABLE APPROACH: Conservative quality calculation with gradual adaptation
        """
        
        # Get base quality from parent class (already has good fixes)
        base_quality = super().calculate_quality(
            loss_before, loss_after, data_sizes, param_update,
            round_num, client_id, all_loss_improvements
        )
        
        # PHASE 1: Pure exploration (rounds 0 to initial_rounds)
        if round_num < self.initial_rounds:
            # Almost uniform quality during exploration
            exploration_quality = 0.4 + 0.2 * np.random.random()  # Random between 0.4-0.6
            
            # self.logger.info(f"R{round_num} Client {client_id}: EXPLORATION MODE "
            #                f"({round_num}/{self.initial_rounds}), quality={exploration_quality:.4f}")
            # return exploration_quality
            
        # PHASE 2: Gradual transition (rounds initial_rounds to initial_rounds*2)
        elif round_num < self.initial_rounds * 2:
            transition_rounds = self.initial_rounds  # Duration of transition phase
            transition_weight = (round_num - self.initial_rounds) / transition_rounds  # 0 to 1
            exploration_component = (1 - transition_weight) * 0.5
            quality_component = transition_weight * base_quality
            
            final_quality = exploration_component + quality_component
            
            # self.logger.info(f"R{round_num} Client {client_id}: TRANSITION MODE, "
            #                f"weight={transition_weight:.3f}, "
            #                f"final={final_quality:.4f}")
            
        # PHASE 3: Full quality-based selection (rounds initial_rounds*2+)
        else:
            # Use base quality but with conservative bounds
            final_quality = max(self.baseline_quality, min(0.9, base_quality))
            
            # Track client participation for diversity
            if client_id not in self.client_participation:
                self.client_participation[client_id] = 0
            self.client_participation[client_id] += 1
            
            # Small diversity bonus for underused clients
            if len(self.client_participation) > 10:
                avg_participation = np.mean(list(self.client_participation.values()))
                client_usage_ratio = self.client_participation[client_id] / avg_participation
                
                # Small bonus for underused clients (max 0.1 bonus)
                if client_usage_ratio < 0.7:
                    diversity_bonus = 0.05 * (0.7 - client_usage_ratio)
                    final_quality = min(0.9, final_quality + diversity_bonus)
                    
            self.logger.info(f"R{round_num} Client {client_id}: QUALITY MODE, "
                           f"base={base_quality:.4f}, final={final_quality:.4f}")
        
        # Track round statistics
        self.round_qualities.append(final_quality)
        
        # Log comprehensive statistics every 10 rounds
        if round_num % 10 == 0 and len(self.round_qualities) > 0:
            recent_qualities = self.round_qualities[-50:] if len(self.round_qualities) >= 50 else self.round_qualities
            
            self.logger.info(f"=== ROUND {round_num} STATISTICS ===")
            self.logger.info(f"Recent quality stats: "
                           f"mean={np.mean(recent_qualities):.4f}, "
                           f"std={np.std(recent_qualities):.4f}, "
                           f"min={np.min(recent_qualities):.4f}, "
                           f"max={np.max(recent_qualities):.4f}")
            
            if len(self.client_participation) > 0:
                participation_values = list(self.client_participation.values())
                self.logger.info(f"Client participation: "
                               f"unique_clients={len(self.client_participation)}, "
                               f"avg_selections={np.mean(participation_values):.1f}, "
                               f"selection_std={np.std(participation_values):.1f}")
                
        return final_quality
    
    def get_phase_info(self, round_num):
        """Helper to understand which phase we're in"""
        if round_num < self.initial_rounds:
            return "EXPLORATION", f"Pure random selection for diversity (round {round_num}/{self.initial_rounds})"
        elif round_num < self.initial_rounds * 2:
            weight = (round_num - self.initial_rounds) / self.initial_rounds
            return "TRANSITION", f"Mixing random ({100*(1-weight):.0f}%) + quality ({100*weight:.0f}%)"
        else:
            return "QUALITY_BASED", f"Full quality-based selection with diversity bonus"

