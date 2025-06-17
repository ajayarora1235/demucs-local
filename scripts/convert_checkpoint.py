"""
Convert the demucs_with_stft_istft.th checkpoint from 2D to 1D weights.
"""
import torch
from weight_converter import convert_2d_to_1d_weights

def main():
    checkpoint_path = "checkpoints/demucs_with_stft_istft.th"
    output_path = "checkpoints/demucs_with_stft_istft_1d_conv_only.th"
    
    print("Converting checkpoint from 2D to 1D convolutions...")
    print(f"Input: {checkpoint_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    # Load original checkpoint
    print("Loading original checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Print checkpoint structure
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Get state dict (handle Demucs format)
    if 'state' in checkpoint:
        state_dict = checkpoint['state']
        print("Found 'state' key in checkpoint (Demucs format)")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Found 'state_dict' key in checkpoint")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("Found 'model' key in checkpoint")
    else:
        state_dict = checkpoint
        print("Using checkpoint directly as state_dict")
    
    print(f"Number of parameters in original model: {len(state_dict)}")
    
    # Print some example keys to understand structure
    print("Sample parameter keys:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        param = state_dict[key]
        if hasattr(param, 'shape'):
            print(f"  {key}: {param.shape}")
        else:
            print(f"  {key}: {type(param)}")
    
    # Convert weights
    print("\nConverting weights...")
    converted_state_dict = convert_2d_to_1d_weights(state_dict)
    
    print(f"Number of parameters in converted model: {len(converted_state_dict)}")
    
    # Update checkpoint with converted weights
    if 'state' in checkpoint:
        checkpoint['state'] = converted_state_dict
    elif 'state_dict' in checkpoint:
        checkpoint['state_dict'] = converted_state_dict
    elif 'model' in checkpoint:
        checkpoint['model'] = converted_state_dict
    else:
        checkpoint = converted_state_dict
    
    # Save converted checkpoint
    print(f"\nSaving converted checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    
    print("✅ Conversion complete!")
    
    # Verify the conversion worked
    print("\nVerifying conversion...")
    converted_checkpoint = torch.load(output_path, map_location='cpu')
    converted_state = converted_checkpoint.get('state', converted_checkpoint)
    
    print("Sample converted parameter shapes:")
    for i, key in enumerate(list(converted_state.keys())[:10]):
        param = converted_state[key]
        if hasattr(param, 'shape'):
            print(f"  {key}: {param.shape}")

if __name__ == "__main__":
    main() 