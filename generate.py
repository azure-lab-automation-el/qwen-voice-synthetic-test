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

def compare_variants():
    """Rank synthetic-silver Hebrew samples; ASR cannot establish accent quality."""
    from faster_whisper import WhisperModel
    from difflib import SequenceMatcher
    import subprocess
    from pathlib import Path
    import soundfile as sf
    root = Path('hebrew-output'); root.mkdir(exist_ok=True)
    reference = Path('silver.wav')
    # The short reference keeps the same synthetic speaker while weakening the English ICL context.
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(reference),'-t','2.7','-af','afade=t=out:st=2.45:d=0.25',str(root/'silver-short.wav')],check=True)
    # Different reference modes test whether the English-language ICL codes cause the accent.
    subprocess.run(['qwentts/build/qwen-codec','--model','models/codec.gguf','--talker','models/talker.gguf','-i',str(root/'silver-short.wav')],check=True,stdout=subprocess.DEVNULL)
    target = 'שלום, מה שלומך היום?'
    prompts = [
      ('a','ʃalˈom, mˈa ʃlomχˈa hajˈom?'),
      ('b','ʃaˈlom, ma ʃloˈmχa haˈjom?'),
      ('c','ʃalˈom. ma ʃlomˈχa haˈjom?'),
    ]
    specs = [
      ('full-greedy',['--ref-wav','silver.wav','--ref-text','silver.txt','--greedy']),
      ('short-greedy',['--ref-wav',str(root/'silver-short.wav'),'--greedy']),
      ('short-temp',['--ref-wav',str(root/'silver-short.wav'),'--temp','0.55','--sub-temp','0.55','--seed','42']),
      ('speaker-only',['--ref-spk',str(root/'silver-short.spk'),'--greedy']),
    ]
    asr = WhisperModel('small',device='cpu',compute_type='int8')
    results=[]
    for pname,prompt in prompts:
      for sname,flags in specs:
        key=f'{pname}-{sname}'; out=root/f'{key}.wav'
        cmd=['qwentts/build/qwen-tts','--model','models/talker.gguf','--codec','models/codec.gguf',*flags,'--lang','auto','-o',str(out)]
        print('⏳ יוצרת גרסה',key,flush=True)
        try:
          subprocess.run(cmd,input=prompt,text=True,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=240)
          audio,sr=sf.read(out)
          if not 1<len(audio)/sr<20: raise ValueError('Unexpected duration')
          segs,_=asr.transcribe(str(out),language='he',beam_size=5)
          heard=' '.join(seg.text.strip() for seg in segs)
          similarity=SequenceMatcher(None,target.replace(' ','').replace('?',''),heard.replace(' ','').replace('?','')).ratio()
          results.append((round(similarity,3),key,heard,round(len(audio)/sr,2)))
          print('✅',key,heard,'התאמת תמלול',round(similarity,3),flush=True)
        except Exception as exc:
          print('❌',key,repr(exc),getattr(exc,'stderr',b''),flush=True)
    results.sort(reverse=True)
    lines=['| דירוג | גרסה | התאמת תמלול (לא דירוג מבטא) | תמלול | שניות |','|---|---|---:|---|---:|']
    for rank,(score,key,heard,duration) in enumerate(results,1):
      lines.append(f'| {rank} | {key} | {score:.3f} | {heard} | {duration} |')
    text='\n'.join(lines)+'\n'
    (root/'results.md').write_text(text,encoding='utf-8')
    print(text)
    for _,key,_,_ in results[:3]:
      subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(root/f'{key}.wav'),'-c:a','libopus','-b:a','32k',str(root/f'{key}.ogg')],check=True)
    if not results: raise RuntimeError('No usable samples generated')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('task',choices=['references','check-hebrew','compare-variants'],nargs='?',default='references')
    args=parser.parse_args()
    {'references': references, 'check-hebrew': check_hebrew, 'compare-variants': compare_variants}[args.task]()
