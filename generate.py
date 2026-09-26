"""Transient synthetic references and Hebrew sample check; no owner data."""
import argparse
import gc
from pathlib import Path
import soundfile as sf

TEXT = 'The quiet signal was hidden beneath the noise. Now we know where to look.'
VOICES = {
    'alto': 'An original fictional female narrator. Low velvety alto, intimate and mysterious, slow measured pace, crisp articulation, faint breathiness. Do not imitate a real person.',
    'smoky': 'An original fictional female narrator. Mature smoky mezzo, soft hushed suspense, grounded and warm, restrained dramatic rises, clear articulation. Do not imitate a real person.',
    'silver': 'An original fictional female narrator. Ethereal silver soprano, quiet wonder and subtle mystery, rounded vowels, deliberate gentle rhythm, natural not robotic. Do not imitate a real person.',
}

def references():
    import torch
    from qwen_tts import Qwen3TTSModel
    out=Path('synthetic-references');out.mkdir(exist_ok=True)
    model=Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign',device_map='cpu',dtype=torch.float32,attn_implementation='sdpa')
    for name,instruct in VOICES.items():
        with torch.inference_mode():
            audio,sr=model.generate_voice_design(text=TEXT,language='English',instruct=instruct,non_streaming_mode=True,max_new_tokens=2048)
        sf.write(out/f'{name}.wav',audio[0],sr)
        print(name,round(len(audio[0])/sr,2),'seconds')
        del audio;gc.collect()

def check_hebrew():
    from faster_whisper import WhisperModel
    p=Path('hebrew-output/silver-hebrew.wav')
    audio,sr=sf.read(p)
    duration=len(audio)/sr
    print('Hebrew WAV:',sr,'Hz,',round(duration,2),'s')
    if not 1 < duration < 20: raise RuntimeError('Output length outside expected range')
    model=WhisperModel('small',device='cpu',compute_type='int8')
    segments, info=model.transcribe(str(p),language='he',beam_size=5)
    text=' '.join(s.text.strip() for s in segments)
    print('Hebrew ASR:',text,'language probability:',round(info.language_probability,3))
    Path('hebrew-output/asr.txt').write_text(text+'\n',encoding='utf-8')
    # ASR is a screen, not proof of intelligibility. Parent listens before user delivery.

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('task',choices=['references','check-hebrew'],nargs='?',default='references')
    args=parser.parse_args()
    (references if args.task=='references' else check_hebrew)()
