import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("C:/Users/admin/Desktop/recognition speech/wav2vec2-darija-processor/kaggle/input/process/wav2vec2-darija-processor")
model = Wav2Vec2ForCTC.from_pretrained("C:/Users/admin/Desktop/recognition speech/wav2vec2-darija-processor/kaggle/input/process/wav2vec2-darija")

def transcribe_darija(audio_path):
    # Charger et resampler audio à 16 kHz
    speech, sample_rate = librosa.load(audio_path, sr=16000)
    print(f"Sample rate après resampling: {sample_rate}, samples: {len(speech)}, duration (s): {len(speech)/sample_rate:.2f}")
    
    # Padding si audio trop court (ici 1 sec = 16000 samples)
    min_length = 16000
    if len(speech) < min_length:
        pad_length = min_length - len(speech)
        speech = np.pad(speech, (0, pad_length), 'constant', constant_values=0)
        print(f"Audio paded to {len(speech)} samples.")

    input_values = processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

