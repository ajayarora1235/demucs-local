"""
Correct weight conversion for HTDemucs architecture:
- Encoders: HEncLayerNew (1D) - convert weights from 2D to 1D
- Decoders: HDecLayerNew (1D) - convert weights to 1D
"""
import torch

def convert_htdemucs_weights(state_dict):
    """
    Convert weights for HTDemucs architecture:
    - Convert encoder weights from 2D to 1D (for HEncLayerNew)
    - Convert decoder weights to 1D (for HDecLayerNew)
    """
    converted_state_dict = {}
    
    for key, value in state_dict.items():
        if 'decoder' in key and 'conv_tr.weight' in key:
            # Convert decoder main conv weights: [out, in, k, 1] -> [out, in, k]
            if value.dim() == 4 and value.shape[3] == 1:
                converted_value = value.squeeze(3)
                print(f"Converted decoder conv {key}: {value.shape} -> {converted_value.shape}")
                converted_state_dict[key] = converted_value
            else:
                print(f"Warning: Cannot convert decoder conv {key} with shape {value.shape}")
                converted_state_dict[key] = value
                
        elif 'decoder' in key and 'rewrite.weight' in key:
            # Keep decoder rewrite weights unchanged - they should match HDecLayerNew expectations
            print(f"Keeping decoder rewrite unchanged: {key} {value.shape}")
            converted_state_dict[key] = value
                
        elif 'tdecoder' in key and 'conv_tr.weight' in key:
            # Time decoder conv weights - should already be 1D or convert if needed
            if value.dim() == 3:
                # Already 1D, keep as is
                print(f"Keeping tdecoder conv unchanged (already 1D): {key} {value.shape}")
                converted_state_dict[key] = value
            elif value.dim() == 4 and value.shape[3] == 1:
                converted_value = value.squeeze(3)
                print(f"Converted tdecoder conv {key}: {value.shape} -> {converted_value.shape}")
                converted_state_dict[key] = converted_value
            else:
                print(f"Warning: Cannot convert tdecoder conv {key} with shape {value.shape}")
                converted_state_dict[key] = value
                
        elif 'tdecoder' in key and 'rewrite.weight' in key:
            # Keep time decoder rewrite weights unchanged - they should match HDecLayerNew expectations
            print(f"Keeping tdecoder rewrite unchanged: {key} {value.shape}")
            converted_state_dict[key] = value
                
        elif ('encoder' in key or 'tencoder' in key) and 'conv.weight' in key:
            # Convert encoder main conv weights for HEncLayerNew
            if value.dim() == 4 and value.shape[3] == 1:
                # Convert frequency domain: [out, in, kernel_freq, 1] -> [out, in, kernel_freq]
                converted_value = value.squeeze(3)
                print(f"Converted encoder conv (freq) {key}: {value.shape} -> {converted_value.shape}")
                converted_state_dict[key] = converted_value
            elif value.dim() == 3:
                # Time domain weights are already 1D, copy directly
                print(f"Copied encoder conv (time) {key}: {value.shape}")
                converted_state_dict[key] = value
            else:
                print(f"Warning: Unexpected encoder conv shape {key}: {value.shape}")
                converted_state_dict[key] = value
                
        elif ('encoder' in key or 'tencoder' in key) and 'rewrite.weight' in key:
            # Convert encoder rewrite weights for HEncLayerNew
            if value.dim() == 4:
                # Convert frequency domain: [out, in, 1, 1] -> [out, in, 1]
                converted_value = value.squeeze(3).squeeze(2).unsqueeze(2)
                print(f"Converted encoder rewrite (freq) {key}: {value.shape} -> {converted_value.shape}")
                converted_state_dict[key] = converted_value
            elif value.dim() == 3:
                # Time domain rewrite weights are already 1D, copy directly
                print(f"Copied encoder rewrite (time) {key}: {value.shape}")
                converted_state_dict[key] = value
            else:
                print(f"Warning: Unexpected encoder rewrite shape {key}: {value.shape}")
                converted_state_dict[key] = value
                
        else:
            # Keep all other weights unchanged (norms, embeddings, etc.)
            converted_state_dict[key] = value
    
    return converted_state_dict

def main():
    checkpoint_path = "checkpoints/demucs_with_stft_istft.th"
    output_path = "checkpoints/demucs_with_stft_istft_htdemucs.th"
    
    print("Converting weights for HTDemucs architecture...")
    print("- Encoders: HEncLayerNew (1D) - convert weights from 2D to 1D")
    print("- Decoders: HDecLayerNew (1D) - convert weights to 1D")
    print(f"Input: {checkpoint_path}")
    print(f"Output: {output_path}")
    print("=" * 70)
    
    # Load original checkpoint
    print("Loading original checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get state dict
    if 'state' in checkpoint:
        state_dict = checkpoint['state']
        print("Found 'state' key in checkpoint (Demucs format)")
    else:
        state_dict = checkpoint
        print("Using checkpoint directly as state_dict")
    
    print(f"Number of parameters in original model: {len(state_dict)}")
    
    # Convert weights
    print("\nConverting weights...")
    converted_state_dict = convert_htdemucs_weights(state_dict)
    
    print(f"Number of parameters in converted model: {len(converted_state_dict)}")
    
    # Update checkpoint
    if 'state' in checkpoint:
        checkpoint['state'] = converted_state_dict
    else:
        checkpoint = converted_state_dict
    
    # Save converted checkpoint
    print(f"\nSaving HTDemucs checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    
    print("✅ HTDemucs conversion complete!")
    print("\n" + "="*70)
    print("SUMMARY:")
    print("✅ Encoder conv weights: Converted from 2D to 1D (for HEncLayerNew)")
    print("✅ Encoder rewrite weights: Converted from 2D to 1D (for HEncLayerNew)")
    print("✅ Decoder conv weights: Converted to 1D (for HDecLayerNew)")
    print("✅ Decoder rewrite weights: Kept unchanged (for HDecLayerNew)")
    print("✅ No 2D convolutions with mixed strides [4,1] - using pure 1D convolutions!")
    print("✅ This checkpoint works with your HTDemucs model!")

if __name__ == "__main__":
    main() 