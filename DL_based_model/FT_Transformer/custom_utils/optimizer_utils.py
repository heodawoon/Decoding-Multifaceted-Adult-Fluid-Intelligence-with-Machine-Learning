import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from omegaconf import DictConfig


def get_parameter_groups(model: nn.Module, config: DictConfig) -> List[Dict[str, Any]]:
    """
    Create parameter groups with different learning rates and weight decay for different parameter types.
    
    Args:
        model: PyTorch model
        config: Configuration containing optimizer settings
        
    Returns:
        List of parameter group dictionaries
    """
    # Define parameter grouping rules
    def needs_weight_decay(name: str) -> bool:
        """Parameters that should have weight decay applied."""
        return all(x not in name for x in ['tokenizer', '.norm', '.bias', '.embedding'])
    
    def is_embedding_param(name: str) -> bool:
        """Check if parameter is an embedding layer."""
        return 'embedding' in name.lower()
    
    def is_attention_param(name: str) -> bool:
        """Check if parameter is from attention layers."""
        return any(x in name.lower() for x in ['attention', 'attn', 'q_', 'k_', 'v_'])
    
    def is_ffn_param(name: str) -> bool:
        """Check if parameter is from feed-forward network."""
        return any(x in name.lower() for x in ['linear', 'ffn', 'mlp'])
    
    # Group parameters
    embedding_params = []
    attention_params = []
    ffn_params = []
    other_params_with_wd = []
    params_without_wd = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if is_embedding_param(name):
            embedding_params.append(param)
        elif is_attention_param(name):
            attention_params.append(param)
        elif is_ffn_param(name):
            ffn_params.append(param)
        elif needs_weight_decay(name):
            other_params_with_wd.append(param)
        else:
            params_without_wd.append(param)
    
    # Create parameter groups with different learning rates
    param_groups = []
    
    # Embedding parameters (usually need lower learning rate)
    if embedding_params:
        param_groups.append({
            'params': embedding_params,
            'lr': config.optimizer.lr * getattr(config.optimizer, 'embedding_lr_mult', 0.1),
            'weight_decay': config.optimizer.weight_decay * getattr(config.optimizer, 'embedding_wd_mult', 0.1),
            'name': 'embeddings'
        })
    
    # Attention parameters
    if attention_params:
        param_groups.append({
            'params': attention_params,
            'lr': config.optimizer.lr * getattr(config.optimizer, 'attention_lr_mult', 1.0),
            'weight_decay': config.optimizer.weight_decay,
            'name': 'attention'
        })
    
    # FFN parameters
    if ffn_params:
        param_groups.append({
            'params': ffn_params,
            'lr': config.optimizer.lr * getattr(config.optimizer, 'ffn_lr_mult', 1.0),
            'weight_decay': config.optimizer.weight_decay,
            'name': 'ffn'
        })
    
    # Other parameters with weight decay
    if other_params_with_wd:
        param_groups.append({
            'params': other_params_with_wd,
            'lr': config.optimizer.lr,
            'weight_decay': config.optimizer.weight_decay,
            'name': 'other_with_wd'
        })
    
    # Parameters without weight decay (bias, norm layers, etc.)
    if params_without_wd:
        param_groups.append({
            'params': params_without_wd,
            'lr': config.optimizer.lr,
            'weight_decay': 0.0,
            'name': 'no_weight_decay'
        })
    
    return param_groups


def create_optimizer(model: nn.Module, config: DictConfig) -> torch.optim.Optimizer:
    """
    Create optimizer with sophisticated parameter grouping.
    
    Args:
        model: PyTorch model
        config: Configuration containing optimizer settings
        
    Returns:
        Configured optimizer
    """
    param_groups = get_parameter_groups(model, config)
    
    # Print parameter group information
    print("Parameter Groups:")
    for i, group in enumerate(param_groups):
        num_params = sum(p.numel() for p in group['params'])
        print(f"  Group {i+1} ({group['name']}): {num_params:,} parameters, "
              f"lr={group['lr']:.2e}, wd={group['weight_decay']:.2e}")
    
    # Create optimizer based on config
    optimizer_name = config.optimizer.name.lower()
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=getattr(config.optimizer, 'betas', (0.9, 0.999)),
            eps=getattr(config.optimizer, 'eps', 1e-8)
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=getattr(config.optimizer, 'betas', (0.9, 0.999)),
            eps=getattr(config.optimizer, 'eps', 1e-8)
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            momentum=getattr(config.optimizer, 'momentum', 0.9),
            nesterov=getattr(config.optimizer, 'nesterov', False)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def get_parameter_stats(model: nn.Module) -> Dict[str, int]:
    """Compute statistics of model parameters, including total, trainable, frozen,
    and per-layer parameter counts."""
    total_params = 0
    trainable_params = 0
    param_stats = {}
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
            
        # Group by layer type
        layer_type = name.split('.')[0] if '.' in name else 'other'
        param_stats[layer_type] = param_stats.get(layer_type, 0) + num_params
    
    param_stats['total'] = total_params
    param_stats['trainable'] = trainable_params
    param_stats['frozen'] = total_params - trainable_params
    
    return param_stats
