"""Training script for baseline diffusion Latin square model."""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
import wandb

from pl_latin_model import LatinSquareModel


def main():
    # Set up arguments
    class Args:
        def __init__(self):
            # Model parameters
            self.diffusion_type = 'categorical'
            self.diffusion_schedule = 'linear'
            self.diffusion_steps = 1000
            self.inference_diffusion_steps = 50  # Faster inference for testing
            self.inference_schedule = 'linear'
            self.inference_trick = 'ddim'
            
            # Architecture
            self.n_layers = 6  # Smaller for faster training
            self.hidden_dim = 128  # Smaller for faster training
            self.sparse_factor = -1
            self.aggregation = 'sum'
            self.use_activation_checkpoint = False
            
            # Training
            self.batch_size = 16  # Smaller batch size
            self.learning_rate = 1e-4
            self.weight_decay = 0.0
            self.lr_scheduler = 'constant'
            self.num_workers = 2
            self.fp16 = False
            self.sequential_sampling = 1
            self.parallel_sampling = 1
            
            # Training control
            self.num_epochs = 10  # Start with fewer epochs
            self.project_name = 'latin_squares_baseline'
            self.wandb_entity = None
            self.wandb_logger_name = 'baseline_diffusion'
    
    args = Args()
    
    # Initialize model
    print("Initializing Latin Square model...")
    model = LatinSquareModel(param_args=args)
    
    # Set up logging
    wandb_id = wandb.util.generate_id()
    wandb_logger = WandbLogger(
        name=args.wandb_logger_name,
        project=args.project_name,
        entity=args.wandb_entity,
        save_dir='./logs',
        id=wandb_id,
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val/avg_orthogonality',
        mode='max',
        save_top_k=3,
        save_last=True,
        dirpath=f'./checkpoints/{args.wandb_logger_name}',
        filename='baseline-{epoch:02d}-{val/avg_orthogonality:.4f}'
    )
    
    lr_callback = LearningRateMonitor(logging_interval='step')
    
    # Set up trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=args.num_epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback],
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        precision=16 if args.fp16 else 32,
        gradient_clip_val=1.0,  # Add gradient clipping
    )
    
    print(f"Model architecture:\n{model.model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output dimension: {model.output_dim}")
    
    # Train the model
    print("Starting training...")
    trainer.fit(model)
    
    # Test the best model
    print("Testing best model...")
    trainer.test(ckpt_path=checkpoint_callback.best_model_path)
    
    # Log final results
    print(f"Training completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    
    # Save a quick evaluation
    model = LatinSquareModel.load_from_checkpoint(
        checkpoint_callback.best_model_path, 
        param_args=args
    )
    model.eval()
    
    # Generate a few samples for inspection
    print("\nGenerating sample Latin squares...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    with torch.no_grad():
        samples = model.generate_samples(batch_size=5, device=device)
        
        for i in range(5):
            squares = samples[i].reshape(3, 10, 10).cpu().numpy()
            print(f"\nSample {i+1}:")
            
            for j, square in enumerate(squares):
                print(f"Square {j+1}:")
                print(square)
                
                # Check validity
                from orthogonality import verify_latin_square
                is_valid = verify_latin_square(square)
                print(f"Valid Latin square: {is_valid}")
            
            # Check orthogonality
            from orthogonality import orthogonality
            orth_score = orthogonality(squares[0], squares[1], squares[2])
            print(f"Orthogonality score: {orth_score:.4f}")
            print("-" * 50)


if __name__ == '__main__':
    main()