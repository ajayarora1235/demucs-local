import torch
import coremltools as ct
import numpy as np
import librosa
import torch.nn as nn
import argparse
from typing import Tuple, Any
import soundfile as sf
import os

from utils.settings import get_model_from_config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert model to CoreML format')
    parser.add_argument('--config_path', type=str, required=True,
                      help='Path to the model config file')
    parser.add_argument('--start_check_point', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                      help='Type of model (e.g., bs_roformer)')
    parser.add_argument('--use_stft', action='store_true',
                      help='Whether to use STFT preprocessing')
    parser.add_argument('--example_audio', type=str, required=True,
                      help='Path to example audio file for conversion')
    parser.add_argument('--output_path', type=str, default='model.mlpackage',
                      help='Path to save the CoreML model')
    parser.add_argument('--half', action='store_true',
                      help='Whether to convert model to half precision')
    return parser.parse_args()

def load_model_and_config(model_type: str, config_path: str, checkpoint_path: str, 
                         use_stft: bool = False, half: bool = False) -> Tuple[torch.nn.Module, Any]:
    """Load model and config, optionally filtering STFT weights."""
    model, config = get_model_from_config(model_type, config_path)
    
    device = 'cpu'
    
    if model_type in ['htdemucs', 'apollo']:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Fix for htdemucs pretrained models
        if 'state' in state_dict:
            state_dict = state_dict['state']
        # Fix for apollo pretrained models
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']


        ## filter out padding_zero
        state_dict = {k: v for k, v in state_dict.items() 
                     if 'padding_zero' not in k}
    else:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    
    if use_stft:
        # Filter out STFT/ISTFT related weights so we do it ourselves prior to input
        stft_related_keys = [
            'stft_window_fn',
            'stft_kwargs',
            'multi_stft_resolution_loss_weight',
            'multi_stft_resolutions_window_sizes',
            'multi_stft_n_fft',
            'multi_stft_window_fn',
            'multi_stft_kwargs'
        ]
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                             if not any(stft_key in k for stft_key in stft_related_keys)}
        
        print('found number of keys: ', len(filtered_state_dict.keys()))
        
        # Verify no STFT-related weights remain
        for k in filtered_state_dict.keys():
            assert not any(stft_key in k for stft_key in stft_related_keys), \
                f"Found STFT-related key in filtered weights: {k}"
    else:
        filtered_state_dict = state_dict
    
    model.load_state_dict(filtered_state_dict)
    model.eval()
    model = model.half() if half else model

    return model, config

def prepare_audio(audio_path: str, config: Any, half: bool = False) -> torch.Tensor:
    """Load and prepare audio for model input."""
    sample_rate = getattr(config.audio, 'sample_rate', 44100)
    print(f"Processing track: {audio_path}")
    
    try:
        mix, sr = librosa.load(audio_path, sr=sample_rate, mono=False)
    except Exception as e:
        print(f'Cannot read track: {audio_path}')
        print(f'Error message: {str(e)}')
        exit()

    # Handle mono audio
    if len(mix.shape) == 1:
        mix = np.expand_dims(mix, axis=0)
        if 'num_channels' in config.audio:
            if config.audio['num_channels'] == 2:
                print(f'Convert mono track to stereo...')
                mix = np.concatenate([mix, mix], axis=0)

    return torch.from_numpy(mix).half() if half else torch.from_numpy(mix)

def retrieve_audio_chunk(audio: torch.Tensor, config: Any, half: bool = False) -> torch.Tensor:
    """Retrieve audio chunk with appropriate padding to be input into model."""
    chunk_size = config.audio.chunk_size
    chunk_size = 44100
    num_overlap = config.inference.num_overlap
    step = chunk_size // num_overlap
    border = chunk_size - step
    
    # Convert to float32 for padding operations
    original_dtype = audio.dtype
    audio = audio.float()
    
    # Add padding for generic mode
    if audio.shape[-1] > 2 * border and border > 0:
        audio = nn.functional.pad(audio, (border, border), mode="reflect")
    
    # Extract first chunk
    part = audio[:, :chunk_size]
    chunk_len = part.shape[-1]
    
    if chunk_len > chunk_size // 2:
        part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode="reflect")
    else:
        part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode="constant", value=0)
    
    # Convert back to original dtype
    if half:
        part = part.half()
    else:
        part = part.to(original_dtype)
    
    return part

def compute_stft(audio: torch.Tensor, config: Any, half: bool = False) -> torch.Tensor:
    """Compute STFT of the audio input."""
    n_fft = config.model.stft_n_fft
    hop_length = config.model.stft_hop_length
    win_length = config.model.stft_win_length
    window = torch.hann_window(win_length).half() if half else torch.hann_window(win_length)
    
    # Ensure audio is in float32 for STFT
    audio_float32 = audio.float()
    
    # Compute STFT
    stft_input = torch.stft(
        audio_float32,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )
    
    stft_input = torch.view_as_real(stft_input)
    return stft_input.half() if half else stft_input.contiguous()

def convert_to_coreml(model: torch.nn.Module, example_input: torch.Tensor, 
                     output_path: str) -> None:
    """Convert PyTorch model to CoreML format."""
    model.eval()  # Ensure model is in eval mode
    model_output_path = output_path 
    # model_output_path = "demucs_correct_weightloading.mlpackage"

    # print input stats
    print(f"Example input shape: {example_input.shape}")
    print(f"Example input min: {example_input.min():.6f}")
    print(f"Example input max: {example_input.max():.6f}")
    print(f"Example input mean: {example_input.mean():.6f}")
    print(f"Example input non-zero elements: {torch.count_nonzero(example_input)}")

    with torch.no_grad():
        print(f"Example input shape: {example_input.shape}")
        original_output = model(example_input)
        print(f"Original model output stats:")
        print(f"  Shape: {original_output.shape}")
        print(f"  Min: {original_output.min():.6f}")
        print(f"  Max: {original_output.max():.6f}")
        print(f"  Mean: {original_output.mean():.6f}")
        print(f"  Non-zero elements: {torch.count_nonzero(original_output)}")

    #     ## separate into 4 stems
    #     drums = original_output[:, 0]
    #     bass = original_output[:, 1]
    #     other = original_output[:, 2]
    #     vocals = original_output[:, 3]

    #     ## save each stem to a separate file

    #     output_path = os.path.join("coreml_outputs", f"original_output_vocals.wav")
    #     sf.write(output_path, vocals.squeeze(0).T, 44100, subtype='PCM_16')

    #     output_path = os.path.join("coreml_outputs", f"original_output_drums.wav")
    #     sf.write(output_path, drums.squeeze(0).T, 44100, subtype='PCM_16')

    #     output_path = os.path.join("coreml_outputs", f"original_output_bass.wav")
    #     sf.write(output_path, bass.squeeze(0).T, 44100, subtype='PCM_16')

    #     output_path = os.path.join("coreml_outputs", f"original_output_other.wav")
    #     sf.write(output_path, other.squeeze(0).T, 44100, subtype='PCM_16')
        
    

    # #exported_program = torch.jit.trace(model, example_input)
    exported_program = torch.jit.trace(model, example_input, check_trace=False)

    # # Test the traced model
    with torch.no_grad():
        traced_output = exported_program(example_input)
        print(f"\nTraced model output stats:")
        print(f"  Shape: {traced_output.shape}")
        print(f"  Min: {traced_output.min():.6f}")
        print(f"  Max: {traced_output.max():.6f}")
        print(f"  Mean: {traced_output.mean():.6f}")
        print(f"  Non-zero elements: {torch.count_nonzero(traced_output)}")

    #     ## separate into 4 stems
    #     drums = traced_output[:, 0]
    #     bass = traced_output[:, 1]
    #     other = traced_output[:, 2]
    #     vocals = traced_output[:, 3]

    #     output_path = os.path.join("coreml_outputs", f"traced_output_vocals.wav")
    #     sf.write(output_path, vocals.squeeze(0).T, 44100, subtype='PCM_16')

    #     output_path = os.path.join("coreml_outputs", f"traced_output_drums.wav")
    #     sf.write(output_path, drums.squeeze(0).T, 44100, subtype='PCM_16')
        
    #     output_path = os.path.join("coreml_outputs", f"traced_output_bass.wav")
    #     sf.write(output_path, bass.squeeze(0).T, 44100, subtype='PCM_16')

    #     output_path = os.path.join("coreml_outputs", f"traced_output_other.wav")
    #     sf.write(output_path, other.squeeze(0).T, 44100, subtype='PCM_16')
        
    #     # Compare outputs
    #     diff = torch.abs(original_output - traced_output)
    #     print(f"  Max difference from original: {diff.max():.6f}")

    # # try:
    # #     scripted_model = torch.jit.script(model)
    # #     scripted_output = scripted_model(example_input)
    # #     print("Script method works - use this instead of trace")
    # # except Exception as e:
    # #     print(f"Script method failed: {e}")
    
    model_from_trace = ct.convert(
        exported_program,
        inputs=[ct.TensorType(name="audio_input", 
                         shape=(1, 2, 44100),  # Your exact input shape
                         dtype=np.float32)],
        outputs=[ct.TensorType(name="separated_audio", 
                          dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS18,
        # states=[
        #     ct.StateType(
        #         wrapped_type=ct.TensorType(
        #             shape=(2049, 1, 4096),  # [freqs, 1, nfft]
        #             dtype=np.float16
        #         ),
        #         name="cos_kernel",
        #     ),
        #     ct.StateType(
        #         wrapped_type=ct.TensorType(
        #             shape=(2049, 1, 4096),  # [freqs, 1, nfft]
        #             dtype=np.float16
        #         ),
        #         name="sin_kernel",
        #     ),
        #     ct.StateType(
        #         wrapped_type=ct.TensorType(
        #             shape=(4098, 1, 4096),  # [nfft, nfft]
        #             dtype=np.float16
        #         ),
        #         name="istft_inverse_basis",
        #     ),
        #     ct.StateType(
        #         wrapped_type=ct.TensorType(
        #             shape=(4096 + 512 * (1024 - 2),),  # window_sum_size
        #             dtype=np.float16
        #         ),
        #         name="istft_window_sum_inv",
        #     ),
        # ],
        compute_units=ct.ComputeUnit.CPU_ONLY, 
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        source='pytorch',
    )

    # Test with the same input you used for tracing
    test_input = example_input.numpy()  # Convert to numpy
    
    # Get state tensors from model's state dict
    # state = {
    #     "cos_kernel": state_dict["cos_kernel"].numpy().astype(np.float16),
    #     "sin_kernel": state_dict["sin_kernel"].numpy().astype(np.float16),
    #     "istft_inverse_basis": state_dict["istft_inverse_basis"].numpy().astype(np.float16),
    #     "istft_window_sum_inv": state_dict["istft_window_sum_inv"].numpy().astype(np.float16)
    # }
    
    coreml_output = model_from_trace.predict({"audio_input": test_input})

    print("CoreML output stats:")
    output_array = coreml_output["separated_audio"]  # Adjust key name if different
    print(f"  Shape: {output_array.shape}")
    print(f"  Min: {output_array.min()}")
    print(f"  Max: {output_array.max()}")
    print(f"  Non-zero elements: {np.count_nonzero(output_array)}")

    # ## separate into 4 stems
    # drums = output_array[:, 0]
    # bass = output_array[:, 1]
    # other = output_array[:, 2]
    # vocals = output_array[:, 3]
    
    # output_path = os.path.join("coreml_outputs", f"coreml_output_vocals.wav")
    # sf.write(output_path, vocals.squeeze(0).T, 44100, subtype='PCM_16')

    # output_path = os.path.join("coreml_outputs", f"coreml_output_drums.wav")
    # sf.write(output_path, drums.squeeze(0).T, 44100, subtype='PCM_16')

    # output_path = os.path.join("coreml_outputs", f"coreml_output_bass.wav")
    # sf.write(output_path, bass.squeeze(0).T, 44100, subtype='PCM_16')

    # output_path = os.path.join("coreml_outputs", f"coreml_output_other.wav")
    # sf.write(output_path, other.squeeze(0).T, 44100, subtype='PCM_16')

    print('Model converted to CoreML')
    model_from_trace.save(model_output_path)


    loaded = ct.models.MLModel(model_output_path)

    print(f"Model spec version: {loaded.get_spec().specificationVersion}")
    # print(f"Minimum deployment: {loaded._spec.deploymentTarget}")

    # Test loaded (should explode)
    result2 = loaded.predict({"audio_input": example_input})
    result2 = result2['separated_audio']
    # print(result2.keys())
    print(f"Loaded model output stats:")
    print(f"  Shape: {result2.shape}")
    print(f"  Min: {result2.min():.6f}")
    print(f"  Max: {result2.max():.6f}")
    print(f"  Mean: {result2.mean():.6f}")
    print(f"  Non-zero elements: {np.count_nonzero(result2)}")

    drums = result2[:, 0]
    bass = result2[:, 1]
    other = result2[:, 2]
    vocals = result2[:, 3]

    output_path = os.path.join("coreml_outputs", f"coreml_output_vocals.wav")
    sf.write(output_path, vocals.squeeze(0).T, 44100, subtype='PCM_16')

    output_path = os.path.join("coreml_outputs", f"coreml_output_drums.wav")
    sf.write(output_path, drums.squeeze(0).T, 44100, subtype='PCM_16')

    output_path = os.path.join("coreml_outputs", f"coreml_output_bass.wav")
    sf.write(output_path, bass.squeeze(0).T, 44100, subtype='PCM_16')

    output_path = os.path.join("coreml_outputs", f"coreml_output_other.wav")
    sf.write(output_path, other.squeeze(0).T, 44100, subtype='PCM_16')

    print('Model converted to CoreML')
    model_from_trace.save(model_output_path)

def main():
    args = parse_args()
    
    # Load model and config
    model, config = load_model_and_config(
        args.model_type, 
        args.config_path, 
        args.start_check_point,
        args.use_stft,
        args.half
    )
    
    # Prepare audio
    audio = prepare_audio(args.example_audio, config, args.half)
    processed_audio = retrieve_audio_chunk(audio, config, args.half)
    
    # Prepare model input
    if args.use_stft:
        model_input = compute_stft(processed_audio, config, args.half)
    else:
        model_input = processed_audio.unsqueeze(0).contiguous()
    
    # # Get state dict
    # state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    # if 'state_dict' in state_dict:
    #     state_dict = state_dict['state_dict']
    # if 'state' in state_dict:
    #     state_dict = state_dict['state']
    
    # Convert to CoreML
    convert_to_coreml(model, model_input, args.output_path)

if __name__ == "__main__":
    main()
