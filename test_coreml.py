import torch
import coremltools as ct
import numpy as np
import librosa
import soundfile as sf
import os
from typing import Tuple

def load_audio(audio_path: str, sample_rate: int = 44100) -> Tuple[np.ndarray, int]:
    """Load audio file and return audio data and sample rate."""
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=False)
        return audio, sr
    except Exception as e:
        print(f'Cannot read track: {audio_path}')
        print(f'Error message: {str(e)}')
        exit()

def process_chunks(audio: np.ndarray, model: ct.models.MLModel, chunk_size: int = 65536, num_chunks: int = 10) -> np.ndarray:
    """Process audio in chunks through CoreML model."""
    # Ensure audio is 2D (channels, samples)
    if len(audio.shape) == 1:
        audio = np.expand_dims(audio, axis=0)
    
    # Initialize output array
    num_channels = audio.shape[0]
    total_samples = min(audio.shape[1], chunk_size * num_chunks)
    output = np.zeros((4, 2, total_samples))  # 4 stems: drums, bass, other, vocals
    
    # Process each chunk
    for i in range(0, total_samples, chunk_size):
        # Get current chunk
        end_idx = min(i + chunk_size, total_samples)
        chunk = audio[:, i:end_idx]
        
        # Pad if necessary
        if chunk.shape[1] < chunk_size:
            pad_width = ((0, 0), (0, chunk_size - chunk.shape[1]))
            chunk = np.pad(chunk, pad_width, mode='constant')
        
        # Prepare input for model
        model_input = np.expand_dims(chunk, axis=0)  # Add batch dimension
        
        # Run through model
        result = model.predict({"audio_input": model_input})
        separated_audio = result['separated_audio']
        
        # Add to output
        chunk_length = end_idx - i
        output[:, :, i:end_idx] = separated_audio[0, :, :, :chunk_length]
        
        print(f"Processed chunk {i//chunk_size + 1}/{num_chunks}")
    
    return output

def main():
    # Load CoreML model
    model_path = "demucs_correct_weightloading.mlpackage"
    model = ct.models.MLModel(model_path)
    
    # Create output directory if it doesn't exist
    os.makedirs("coreml_outputs", exist_ok=True)
    
    # Load and process audio
    audio_path = "HYBS_TIP_TOE_short.wav"  # Replace with your audio file path
    audio, sr = load_audio(audio_path)
    
    # Process audio in chunks
    separated_audio = process_chunks(audio, model)
    
    # Save each stem
    stems = ['drums', 'bass', 'other', 'vocals']
    for i, stem in enumerate(stems):
        output_path = os.path.join("coreml_outputs", f"chunked_{stem}.wav")
        sf.write(output_path, separated_audio[i].T, sr, subtype='PCM_16')
        print(f"Saved {stem} to {output_path}")

if __name__ == "__main__":
    main() 