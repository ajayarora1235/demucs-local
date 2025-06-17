"""
Weight conversion utility for converting 2D convolution weights to 1D convolution weights.
This is needed when loading checkpoints from original 2D models into modified 1D models.
"""
import torch
import re

def convert_2d_to_1d_weights(state_dict):
    """
    Convert 2D convolution weights to 1D convolution weights.
    
    Args:
        state_dict: Original state dict with 2D weights
        
    Returns:
        converted_state_dict: State dict with 1D weights
    """
    converted_state_dict = {}
    
    for key, value in state_dict.items():
        if 'conv.weight' in key or 'conv_tr.weight' in key:
            # Handle main convolution weights
            if value.dim() == 4:  # [out_channels, in_channels, kernel_h, kernel_w]
                if value.shape[3] == 1:  # kernel_w == 1, can squeeze
                    converted_value = value.squeeze(3)  # Remove time dimension
                    print(f"Converted {key}: {value.shape} -> {converted_value.shape}")
                    converted_state_dict[key] = converted_value
                else:
                    # This shouldn't happen for frequency domain convolutions
                    print(f"Warning: Cannot convert {key} with shape {value.shape}")
                    converted_state_dict[key] = value
            else:
                converted_state_dict[key] = value
        else:
            # Keep all other weights unchanged
            converted_state_dict[key] = value
    
    return converted_state_dict

def load_converted_weights(model, checkpoint_path):
    """
    Load weights from checkpoint with automatic 2D->1D conversion.
    
    Args:
        model: PyTorch model with 1D convolutions
        checkpoint_path: Path to checkpoint with 2D weights
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict (handle different checkpoint formats)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Convert weights
    print("Converting 2D weights to 1D...")
    converted_state_dict = convert_2d_to_1d_weights(state_dict)
    
    # Load converted weights
    try:
        missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        print("✅ Successfully loaded converted weights!")
        return True
    except Exception as e:
        print(f"❌ Error loading converted weights: {e}")
        return False

# Example usage:
if __name__ == "__main__":
    # Example of how to use this
    print("Weight Converter for 2D->1D Convolution Models")
    print("=" * 50)
    
    # Load and convert a checkpoint
    checkpoint_path = "path/to/your/checkpoint.pth"
    
    # Load original checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    original_state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Convert weights
    converted_state_dict = convert_2d_to_1d_weights(original_state_dict)
    
    # Save converted checkpoint
    converted_checkpoint = checkpoint.copy()
    converted_checkpoint['state_dict'] = converted_state_dict
    
    output_path = checkpoint_path.replace('.pth', '_converted_1d.pth')
    torch.save(converted_checkpoint, output_path)
    print(f"Saved converted checkpoint to: {output_path}") 