import numpy as np
import torch.nn.functional as F
import faiss
import torch
from collections import defaultdict, deque

class MemoryBank:
    def __init__(self, embedding_dim=512, max_memories=1000):
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        
        # Single unified storage system
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.client_data = defaultdict(lambda: {
            'embeddings': [],
            'qualities': [], 
            'rounds': [],
            'quality_history': deque(maxlen=50)
        })
        
        # Legacy compatibility
        self.memories = []
        self.client_quality_history = defaultdict(deque)
        self.client_embeddings = defaultdict(list)
        self.client_qualities = defaultdict(list)
        self.client_rounds = defaultdict(list)
        
        self.client_reliability = defaultdict(float)
        self.client_participation = defaultdict(int)
        self.round_count = 0
        self.global_states = {}

    def add_update(self, client_id, embedding, quality_score, round_num):
        """Add client update to memory bank"""
        # Process embedding once
        if isinstance(embedding, torch.Tensor):
            embedding_np = embedding.cpu().numpy().flatten().astype(np.float32)
        else:
            embedding_np = np.array(embedding).flatten().astype(np.float32)
        
        # Ensure correct dimensions
        if len(embedding_np) != self.embedding_dim:
            if len(embedding_np) < self.embedding_dim:
                padding = np.zeros(self.embedding_dim - len(embedding_np))
                embedding_np = np.concatenate([embedding_np, padding])
            else:
                embedding_np = embedding_np[:self.embedding_dim]
        
        # Store in unified structure
        client_entry = self.client_data[client_id]
        client_entry['embeddings'].append(embedding_np)
        client_entry['qualities'].append(quality_score)
        client_entry['rounds'].append(round_num)
        client_entry['quality_history'].append(quality_score)
        
        # Store in legacy structures for compatibility
        self.memories.append({
            'client_id': client_id,
            'embedding': embedding_np,
            'quality': quality_score,
            'round': round_num
        })
        self.client_quality_history[client_id].append(quality_score)
        self.client_embeddings[client_id].append(embedding_np)
        self.client_qualities[client_id].append(quality_score)
        self.client_rounds[client_id].append(round_num)
        
        # Memory management
        if len(client_entry['embeddings']) > self.max_memories:
            client_entry['embeddings'].pop(0)
            client_entry['qualities'].pop(0)
            client_entry['rounds'].pop(0)
            
        if len(self.memories) > self.max_memories:
            self.memories.pop(0)
            self._rebuild_index()
        
        if len(self.client_embeddings[client_id]) > self.max_memories:
            self.client_embeddings[client_id].pop(0)
            self.client_qualities[client_id].pop(0)
            self.client_rounds[client_id].pop(0)
        
        # Add to FAISS index
        self.index.add(embedding_np.reshape(1, -1))
        self.client_participation[client_id] += 1
        self.update_client_reliability(client_id)

    def update_client_reliability(self, client_id):
        """Update client reliability score"""
        if client_id not in self.client_data or not self.client_data[client_id]['qualities']:
            self.client_reliability[client_id] = 0.1
            return 0.1
        
        # Recent average quality (last 5 rounds)
        recent_qualities = self.client_data[client_id]['qualities'][-5:]
        q_recent = np.mean(recent_qualities)
        
        # Participation bonus
        participation_count = len(self.client_data[client_id]['qualities'])
        reliability = q_recent * np.log(1 + participation_count)
        reliability = max(0.1, min(2.0, reliability))
        
        self.client_reliability[client_id] = reliability
        return reliability

    def get_client_reliability(self, client_id):
        """Get client reliability score"""
        if client_id not in self.client_reliability:
            return 0.1
        return self.client_reliability.get(client_id, 0.1)

    def compute_similarity(self, client_id, current_embedding):
        """Compute similarity between current and historical embeddings"""
        if (client_id not in self.client_data or 
            not self.client_data[client_id]['embeddings']):
            return 0.5
        
        # Convert current embedding to tensor
        if isinstance(current_embedding, np.ndarray):
            curr_emb = torch.from_numpy(current_embedding).float()
        else:
            curr_emb = current_embedding.float()
        
        # Ensure correct shape
        if len(curr_emb.shape) == 2:
            curr_emb = curr_emb.flatten()
        
        # Stack all historical embeddings
        hist_embeddings = np.vstack(self.client_data[client_id]['embeddings'])
        hist_emb_tensor = torch.from_numpy(hist_embeddings).float()
        
        # Vectorized cosine similarity computation
        similarities = F.cosine_similarity(
            curr_emb.unsqueeze(0), 
            hist_emb_tensor, 
            dim=1
        )
        
        # Apply recent weighting
        num_embeddings = len(similarities)
        weights = torch.linspace(0.5, 1.0, num_embeddings)
        weighted_similarity = torch.sum(similarities * weights) / torch.sum(weights)
        
        avg_similarity = weighted_similarity.item()
        
        # Apply diversity penalty if needed
        if avg_similarity > 0.95:
            avg_similarity *= 0.9
        
        return max(0.1, min(1.0, avg_similarity))

    def update_round_count(self):
        """Increment the round counter"""
        self.round_count += 1

    def store_global_state(self, round_num, global_state):
        """Store global model state for a round"""
        if isinstance(global_state, dict):
            stored_state = {}
            for name, param in global_state.items():
                if isinstance(param, torch.Tensor):
                    stored_state[name] = param.cpu().detach().clone()
                else:
                    stored_state[name] = param
            self.global_states[round_num] = stored_state
        else:
            if isinstance(global_state, torch.Tensor):
                self.global_states[round_num] = global_state.cpu().detach().clone()
            else:
                self.global_states[round_num] = global_state

    def _rebuild_index(self):
        """Rebuild FAISS index after memory cleanup"""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        for entry in self.memories:
            emb = entry['embedding'].reshape(1, -1).astype(np.float32)
            self.index.add(emb)

    def get_similar_updates(self, query_embedding, k=5):
        """Search for similar updates using FAISS"""
        if self.index.ntotal == 0:
            return []
        
        # Ensure query embedding is correct format
        if isinstance(query_embedding, torch.Tensor):
            query_np = query_embedding.cpu().numpy()
        else:
            query_np = query_embedding
        
        if len(query_np.shape) == 1:
            query_np = query_np.reshape(1, -1)
        
        # Search FAISS index
        distances, indices = self.index.search(query_np.astype(np.float32), min(k, self.index.ntotal))
        
        # Convert to similarity scores
        similarities = 1.0 / (1.0 + distances[0])
        
        results = []
        for i, (idx, sim) in enumerate(zip(indices[0], similarities)):
            if idx != -1 and i < len(self.memories):
                results.append({
                    'similarity': sim,
                    'client_id': self.memories[idx]['client_id'],
                    'quality': self.memories[idx]['quality'],
                    'round': self.memories[idx]['round']
                })
        
        return results

    def get_top_reliable_clients(self, n=10):
        sorted_clients = sorted(
            self.client_reliability.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [client_id for client_id, _ in sorted_clients[:n]]

    def get_client_statistics(self, client_id):
        if client_id not in self.client_participation:
            return None
        scores = list(self.client_quality_history[client_id])
        stats = {
            'participation_count': self.client_participation[client_id],
            'reliability_score': self.client_reliability[client_id],
            'avg_quality': np.mean(scores) if scores else 0,
            'quality_trend': self.calculate_trend(scores),
            'recent_quality': scores[-1] if scores else 0,
        }
        return stats

    def calculate_trend(self, values, window=5):
        if len(values) < 2:
            return 0.0
        recent = values[-min(window, len(values)):]
        if len(recent) < 2:
            return 0.0
        x = np.arange(len(recent))
        y = np.array(recent)
        return np.polyfit(x, y, 1)[0]