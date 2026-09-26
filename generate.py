"""Temporary synthetic-only reference test. No user data, secrets or voice cloning."""
import gc
from pathlib import Path
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

MODEL = 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
TEXT = 'The quiet signal was hidden beneath the noise. Now we know where to look.'
VOICES = {
    'alto': 'An original fictional female narrator. Low velvety alto, intimate and mysterious, slow measured pace, crisp articulation, faint breathiness. Do not imitate a real person.',
    'smoky': 'An original fictional female narrator. Mature smoky mezzo, soft hushed suspense, grounded and warm, restrained dramatic rises, clear articulation. Do not imitate a real person.',
    'silver': 'An original fictional female narrator. Ethereal silver soprano, quiet wonder and subtle mystery, rounded vowels, deliberate gentle rhythm, natural not robotic. Do not imitate a real person.',
}

out=Path('synthetic-references');out.mkdir(exist_ok=True)
model=Qwen3TTSModel.from_pretrained(MODEL,device_map='cpu',dtype=torch.float32,attn_implementation='sdpa')
for name,instruct in VOICES.items():
    with torch.inference_mode():
        audio,sr=model.generate_voice_design(text=TEXT,language='English',instruct=instruct,non_streaming_mode=True,max_new_tokens=2048)
    sf.write(out/f'{name}.wav',audio[0],sr)
    print(name,round(len(audio[0])/sr,2),'seconds')
    del audio;gc.collect()
