import torch
import os
from precompute_istft_weights import precompute_stft_istft_weights

def add_istft_weights_to_model(model_path, output_path=None, n_fft=4096, hop_length=1024, win_length=None, max_frames=512):
    """
    Load an existing HTDemucs model and add precomputed ISTFT weights to it
    
    Args:
        model_path: Path to the existing .th model file
        output_path: Path to save the updated model (if None, overwrites original)
        n_fft: FFT size (should match model config)
        hop_length: Hop length (should match model config)  
        win_length: Window length (if None, uses n_fft)
        max_frames: Maximum frames to support
    """
    
    if output_path is None:
        output_path = model_path
    
    print(f"Loading model from: {model_path}")
    
    # Load the existing model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Check if it's a full checkpoint or just state dict
    if 'state' in checkpoint:
        state_dict = checkpoint['state']
        print("Found full checkpoint with 'state' key")
    else:
        state_dict = checkpoint
        print("Found state dict directly")
    
    # Precompute STFT and ISTFT weights with the specified parameters
    print(f"\nPrecomputing STFT and ISTFT weights...")
    stft_istft_weights = precompute_stft_istft_weights(
        n_fft=n_fft,
        hop_length=hop_length, 
        win_length=win_length,
        max_frames=max_frames,
        window_type='hann'
    )
    
    # Add the STFT and ISTFT weights to the state dict
    state_dict['istft_inverse_basis'] = stft_istft_weights['istft_inverse_basis']
    state_dict['istft_window_sum_inv'] = stft_istft_weights['istft_window_sum_inv']
    state_dict['cos_kernel'] = stft_istft_weights['cos_kernel']
    state_dict['sin_kernel'] = stft_istft_weights['sin_kernel']
    state_dict['padding_zero'] = stft_istft_weights['padding_zero']
    
    print(f"\nAdded STFT and ISTFT weights to model state dict:")
    print(f"  istft_inverse_basis: {stft_istft_weights['istft_inverse_basis'].shape}")
    print(f"  istft_window_sum_inv: {stft_istft_weights['istft_window_sum_inv'].shape}")
    print(f"  cos_kernel: {stft_istft_weights['cos_kernel'].shape}")
    print(f"  sin_kernel: {stft_istft_weights['sin_kernel'].shape}")
    print(f"  padding_zero: {stft_istft_weights['padding_zero'].shape}")
    
    # Save the updated model
    if 'state' in checkpoint:
        checkpoint['state'] = state_dict
        torch.save(checkpoint, output_path)
    else:
        torch.save(state_dict, output_path)
    
    print(f"\nSaved updated model to: {output_path}")
    
    return output_path

def verify_istft_weights_in_model(model_path):
    """
    Verify that ISTFT weights are present in a model file
    """
    print(f"Verifying ISTFT weights in: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Check if it's a full checkpoint or just state dict
    if 'state' in checkpoint:
        state_dict = checkpoint['state']
    else:
        state_dict = checkpoint
    
    has_inverse_basis = 'istft_inverse_basis' in state_dict
    has_window_sum_inv = 'istft_window_sum_inv' in state_dict
    has_cos_kernel = 'cos_kernel' in state_dict
    has_sin_kernel = 'sin_kernel' in state_dict
    has_padding_zero = 'padding_zero' in state_dict
    
    print(f"istft_inverse_basis present: {has_inverse_basis}")
    if has_inverse_basis:
        print(f"  Shape: {state_dict['istft_inverse_basis'].shape}")
    
    print(f"istft_window_sum_inv present: {has_window_sum_inv}")
    if has_window_sum_inv:
        print(f"  Shape: {state_dict['istft_window_sum_inv'].shape}")
        
    print(f"cos_kernel present: {has_cos_kernel}")
    if has_cos_kernel:
        print(f"  Shape: {state_dict['cos_kernel'].shape}")
        
    print(f"sin_kernel present: {has_sin_kernel}")
    if has_sin_kernel:
        print(f"  Shape: {state_dict['sin_kernel'].shape}")
        
    print(f"padding_zero present: {has_padding_zero}")
    if has_padding_zero:
        print(f"  Shape: {state_dict['padding_zero'].shape}")
    
    if all([has_inverse_basis, has_window_sum_inv, has_cos_kernel, has_sin_kernel, has_padding_zero]):
        print("✅ All STFT and ISTFT weights are present in the model!")
        return True
    else:
        print("❌ Some STFT or ISTFT weights are missing from the model!")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add precomputed ISTFT weights to HTDemucs model")
    parser.add_argument("model_path", help="Path to the HTDemucs .th model file")
    parser.add_argument("--output", help="Output path (default: overwrites input)")
    parser.add_argument("--n_fft", type=int, default=4096, help="FFT size")
    parser.add_argument("--hop_length", type=int, default=1024, help="Hop length")
    parser.add_argument("--win_length", type=int, help="Window length (default: same as n_fft)")
    parser.add_argument("--max_frames", type=int, default=512, help="Maximum frames")
    parser.add_argument("--verify", action="store_true", help="Just verify if weights are present")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_istft_weights_in_model(args.model_path)
    else:
        add_istft_weights_to_model(
            args.model_path,
            args.output,
            args.n_fft,
            args.hop_length,
            args.win_length,
            args.max_frames
        ) 