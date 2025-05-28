"""Lightning module for training the DIFUSCO Latin Square model."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('DIFUSCO/difusco')

from DIFUSCO.difusco.pl_meta_model import COMetaModel
from DIFUSCO.difusco.utils.diffusion_schedulers import InferenceSchedule
from orthogonality import orthogonality, verify_latin_square


class LatinSquareDataset(Dataset):
    """Dataset for Latin squares."""
    
    def __init__(self, data_file, size=10):
        """
        Args:
            data_file: Path to .npy file containing Latin squares
            size: Size of Latin squares (default: 10)
        """
        self.data = np.load(data_file)
        self.size = size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return a single Latin square
        latin_square = self.data[idx]
        # Convert to tensor and add batch dimension for compatibility
        return torch.tensor(latin_square, dtype=torch.long), idx


class LatinSquareModel(COMetaModel):
    """DIFUSCO model for generating three mutually orthogonal Latin squares."""
    
    def __init__(self, param_args=None):
        # Override some parameters for Latin square generation
        if param_args is None:
            param_args = self._get_default_args()
        
        # Set node_feature_only=True since we're working with discrete squares
        super(LatinSquareModel, self).__init__(param_args=param_args, node_feature_only=True)
        
        # Latin square specific parameters
        self.square_size = 10
        self.num_squares = 3  # Generate 3 squares simultaneously
        self.num_symbols = 10  # 0-9 for 10x10 squares
        
        # Override output channels for 3 squares with 10 symbols each
        # Each position needs to output logits for 10 possible symbols
        # Total output: 3 squares * 10*10 positions * 10 symbols = 3000
        self.output_dim = self.num_squares * self.square_size * self.square_size * self.num_symbols
        
        # Create datasets
        self.train_dataset = LatinSquareDataset('latins.npy')
        self.test_dataset = LatinSquareDataset('latins.npy')  # Using same for now
        self.validation_dataset = LatinSquareDataset('latins.npy')  # Using same for now
        
        # Override the model's output layer
        self._setup_output_layer()
        
    def _get_default_args(self):
        """Get default arguments for Latin square model."""
        class Args:
            def __init__(self):
                self.diffusion_type = 'categorical'
                self.diffusion_schedule = 'linear'
                self.diffusion_steps = 1000
                self.inference_diffusion_steps = 1000
                self.inference_schedule = 'linear'
                self.inference_trick = 'ddim'
                self.n_layers = 8
                self.hidden_dim = 256
                self.sparse_factor = -1
                self.aggregation = 'sum'
                self.use_activation_checkpoint = False
                self.batch_size = 32
                self.learning_rate = 1e-4
                self.weight_decay = 0.0
                self.lr_scheduler = 'constant'
                self.num_workers = 4
                self.fp16 = False
                self.sequential_sampling = 1
                self.parallel_sampling = 1
                
        return Args()
    
    def _setup_output_layer(self):
        """Setup the output layer for 3 Latin squares."""
        # Replace the final layer to output the correct dimensions
        # The model should output logits for each position and symbol
        self.model.out_layer = nn.Linear(
            self.model.hidden_dim, 
            self.output_dim
        )
    
    def forward(self, x, t, adj=None, edge_index=None):
        """Forward pass for Latin square generation.
        
        Args:
            x: Input features (batch_size, 3*10*10, feature_dim)
            t: Time steps
            adj: Not used for Latin squares
            edge_index: Not used for Latin squares
            
        Returns:
            Logits for 3 Latin squares (batch_size, 3*10*10*10)
        """
        # For Latin squares, we treat each position as a node
        # x shape: (batch_size, 300) -> (batch_size, 300, 1) for node features
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # Add feature dimension
        
        # Use the sparse forward method since we're treating positions as nodes
        output = self.model.sparse_forward_node_feature_only(x, t, edge_index=None)
        
        return output
    
    def training_step(self, batch, batch_idx):
        """Training step for Latin square generation."""
        latin_squares, _ = batch
        batch_size = latin_squares.shape[0]
        
        # Flatten and replicate to create 3 squares
        # Shape: (batch_size, 10, 10) -> (batch_size, 3, 10, 10)
        target_squares = latin_squares.unsqueeze(1).repeat(1, 3, 1, 1)
        target_squares = target_squares.reshape(batch_size, -1)  # (batch_size, 300)
        
        # Sample time steps
        t = np.random.randint(1, self.diffusion.T + 1, batch_size).astype(int)
        
        # Convert to one-hot for categorical diffusion
        target_onehot = F.one_hot(target_squares.long(), num_classes=self.num_symbols).float()
        # Shape: (batch_size, 300, 10)
        
        # Sample from diffusion process
        xt = self.diffusion.sample(target_onehot, t)
        xt = xt * 2 - 1  # Scale to [-1, 1]
        xt = xt * (1.0 + 0.05 * torch.rand_like(xt))  # Add noise
        
        # Flatten for model input
        xt_flat = xt.reshape(batch_size, -1)  # (batch_size, 3000)
        
        t_tensor = torch.from_numpy(t).float().to(xt.device)
        
        # Forward pass
        logits = self.forward(xt_flat, t_tensor)
        
        # Reshape logits for loss computation
        logits = logits.reshape(batch_size, self.num_squares * self.square_size * self.square_size, self.num_symbols)
        
        # Compute loss
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits.reshape(-1, self.num_symbols), target_squares.reshape(-1).long())
        
        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self.test_step(batch, batch_idx, split='val')
    
    def test_step(self, batch, batch_idx, split='test'):
        """Test step with orthogonality evaluation."""
        latin_squares, batch_indices = batch
        batch_size = latin_squares.shape[0]
        device = latin_squares.device
        
        # Generate samples
        generated_squares = self.generate_samples(batch_size, device)
        
        # Evaluate orthogonality
        orthogonality_scores = []
        valid_squares_count = 0
        
        for i in range(batch_size):
            # Extract the 3 generated squares
            squares = generated_squares[i].reshape(3, self.square_size, self.square_size)
            A, B, C = squares[0], squares[1], squares[2]
            
            # Check if all are valid Latin squares
            valid_A = verify_latin_square(A.cpu().numpy())
            valid_B = verify_latin_square(B.cpu().numpy())
            valid_C = verify_latin_square(C.cpu().numpy())
            
            if valid_A and valid_B and valid_C:
                valid_squares_count += 1
                orth_score = orthogonality(A.cpu().numpy(), B.cpu().numpy(), C.cpu().numpy())
                orthogonality_scores.append(orth_score)
            else:
                orthogonality_scores.append(0.0)  # Invalid squares get 0 orthogonality
        
        avg_orthogonality = np.mean(orthogonality_scores)
        valid_ratio = valid_squares_count / batch_size
        
        metrics = {
            f'{split}/avg_orthogonality': avg_orthogonality,
            f'{split}/valid_squares_ratio': valid_ratio,
            f'{split}/max_orthogonality': np.max(orthogonality_scores) if orthogonality_scores else 0.0
        }
        
        for key, value in metrics.items():
            self.log(key, value)
        
        return metrics
    
    def generate_samples(self, batch_size, device):
        """Generate Latin square samples using diffusion sampling."""
        # Start with random noise
        shape = (batch_size, self.num_squares * self.square_size * self.square_size)
        xt = torch.randn(shape, device=device)
        
        # Convert to categorical (random initial state)
        xt = (xt > 0).long()
        
        # Sampling loop
        inference_steps = self.args.inference_diffusion_steps
        schedule = InferenceSchedule(
            inference_steps, 
            self.diffusion_steps, 
            self.args.inference_schedule
        )
        
        for i in range(inference_steps):
            t = schedule.get_t(i)
            t_array = np.array([t] * batch_size)
            
            # Denoise step
            xt = self.categorical_denoise_step(
                points=None,  # Not used for Latin squares
                xt=xt,
                t=t_array,
                device=device,
                edge_index=None,
                target_t=schedule.get_t(i + 1) if i < inference_steps - 1 else None
            )
        
        return xt
    
    def categorical_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
        """Denoising step for categorical diffusion."""
        # Get model prediction
        t_tensor = torch.from_numpy(t).float().to(device)
        logits = self.forward(xt, t_tensor)
        
        # Reshape logits for categorical sampling
        batch_size = xt.shape[0]
        logits = logits.reshape(batch_size, -1, self.num_symbols)
        
        # Sample from categorical distribution
        probs = F.softmax(logits, dim=-1)
        samples = torch.multinomial(probs.reshape(-1, self.num_symbols), 1)
        samples = samples.reshape(batch_size, -1)
        
        return samples
    
    def train_dataloader(self):
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Get validation dataloader."""
        return DataLoader(
            self.validation_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )