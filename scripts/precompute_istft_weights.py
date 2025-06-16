import torch
import torch.nn.functional as F
import numpy as np
import pickle

def precompute_stft_istft_weights(n_fft=4096, hop_length=1024, win_length=None, max_frames=512, window_type='hann'):
    """
    Precompute ISTFT inverse basis and window sum weights
    
    Args:
        n_fft: FFT size
        hop_length: Hop length for STFT
        win_length: Window length (if None, uses n_fft)
        max_frames: Maximum number of frames to support
        window_type: Type of window ('hann', 'hamming', etc.)
    
    Returns:
        dict: Dictionary containing the precomputed weights
    """
    
    if win_length is None:
        win_length = n_fft
    
    print(f"Precomputing ISTFT weights for:")
    print(f"  n_fft: {n_fft}")
    print(f"  hop_length: {hop_length}")
    print(f"  win_length: {win_length}")
    print(f"  max_frames: {max_frames}")
    print(f"  window_type: {window_type}")
    
    # Create window
    if window_type == 'hann':
        window = torch.hann_window(win_length).float()
    elif window_type == 'hamming':
        window = torch.hamming_window(win_length).float()
    elif window_type == 'blackman':
        window = torch.blackman_window(win_length).float()
    else:
        raise ValueError(f"Unsupported window type: {window_type}")
    
    # Pad window to n_fft if needed
    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        window = F.pad(window, (pad_left, pad_right), mode='constant', value=0)
    
    # Pre-compute fourier basis
    fourier_basis = torch.fft.fft(torch.eye(n_fft, dtype=torch.float32))
    fourier_basis = torch.vstack([
        torch.real(fourier_basis[:n_fft//2 + 1, :]),
        torch.imag(fourier_basis[:n_fft//2 + 1, :])
    ]).float()
    
    # Create inverse basis
    inverse_basis = window * torch.linalg.pinv((fourier_basis * n_fft) / hop_length).T.unsqueeze(1)
    
    # Calculate window sum for overlap-add
    n = n_fft + hop_length * (max_frames - 1)
    window_sum = torch.zeros(n, dtype=torch.float32)
    
    # Get original window and normalize it
    if window_type == 'hann':
        original_window = torch.hann_window(win_length).float()
    elif window_type == 'hamming':
        original_window = torch.hamming_window(win_length).float()
    elif window_type == 'blackman':
        original_window = torch.blackman_window(win_length).float()
    
    window_normalized = original_window / original_window.abs().max()
    
    # Pad the window to n_fft for overlap-add calculation
    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        win_sq = F.pad(window_normalized ** 2, (pad_left, pad_right), mode='constant', value=0)
    else:
        win_sq = window_normalized ** 2
    
    # Calculate overlap-add weights
    for i in range(max_frames):
        sample = i * hop_length
        window_sum[sample: min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    
    # Calculate window sum inverse
    window_sum_inv = n_fft / (window_sum * hop_length + 1e-7)
    
    # Precompute STFT weights (similar to STFT_Process stft_B implementation)
    print("Computing STFT weights...")
    
    # Create time steps and frequency indices
    time_steps = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
    frequencies = torch.arange(n_fft // 2 + 1, dtype=torch.float32).unsqueeze(1)
    
    # Calculate omega matrix
    omega = 2 * torch.pi * frequencies * time_steps / n_fft
    
    # Create cosine and sine kernels for STFT using the same window
    cos_kernel = (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1)
    sin_kernel = (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1)
    
    # Create padding buffer
    padding_zero = torch.zeros((1, 1, n_fft // 2), dtype=torch.float32)
    
    # Return the weights
    weights = {
        'istft_inverse_basis': inverse_basis,
        'istft_window_sum_inv': window_sum_inv,
        'cos_kernel': cos_kernel,
        'sin_kernel': sin_kernel,
        'padding_zero': padding_zero,
        'config': {
            'n_fft': n_fft,
            'hop_length': hop_length,
            'win_length': win_length,
            'max_frames': max_frames,
            'window_type': window_type
        }
    }
    
    print(f"Computed weights:")
    print(f"  istft_inverse_basis shape: {inverse_basis.shape}")
    print(f"  istft_window_sum_inv shape: {window_sum_inv.shape}")
    print(f"  cos_kernel shape: {cos_kernel.shape}")
    print(f"  sin_kernel shape: {sin_kernel.shape}")
    print(f"  padding_zero shape: {padding_zero.shape}")
    
    return weights

def save_istft_weights(weights, filepath):
    """Save the precomputed weights to a file"""
    torch.save(weights, filepath)
    print(f"Saved ISTFT weights to: {filepath}")

def load_istft_weights(filepath):
    """Load the precomputed weights from a file"""
    weights = torch.load(filepath, map_location='cpu')
    print(f"Loaded ISTFT weights from: {filepath}")
    return weights

def create_standard_istft_weights():
    """Create standard ISTFT weights for common configurations"""
    
    # Standard configuration (matches your HTDemucs default)
    standard_config = {
        'n_fft': 4096,
        'hop_length': 1024,  # n_fft // 4
        'win_length': 4096,  # Same as n_fft
        'max_frames': 512,   # Matches your original STFT_Process config
        'window_type': 'hann'
    }
    
    weights = precompute_stft_istft_weights(**standard_config)
    save_istft_weights(weights, 'istft_weights_standard.pt')
    
    return weights

if __name__ == "__main__":
    print("Creating standard ISTFT weights...")
    weights = create_standard_istft_weights()
    
    print("\nTo use these weights in your model:")
    print("1. Load the weights file in your HTDemucs.__init__()")
    print("2. Register the weights as buffers")
    print("3. Use them directly in your _ispec() method") 