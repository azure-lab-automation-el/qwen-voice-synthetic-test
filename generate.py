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
        cmd=['qwen-tts','--model','/opt/models/talker.gguf','--codec','/opt/models/codec.gguf',*flags,'--lang','auto','-o',str(out)]
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

def mysterious_references():
    """Three original fictional female voices derived from the silver design brief."""
    import torch
    from qwen_tts import Qwen3TTSModel
    out=Path('mysterious-references');out.mkdir(exist_ok=True)
    text='There is a small secret in this lesson. Wait for the last page, and I will tell you what it is.'
    designs={
      'velvet':'An original fictional female narrator, not any real person. Keep the sweet, airy, bell-like silver voice. Add quiet mystery: slightly lower pitch, intimate breathy softness, slow measured phrasing, subtle pauses, clear consonants. Natural, never theatrical.',
      'moon':'An original fictional female narrator, not any real person. Sweet silver soprano with a soft, dusky undertone. Whisper-adjacent but fully voiced, thoughtful pauses and a half-smile, calm suspense, clean articulation and a natural rhythm.',
      'secret':'An original fictional female narrator, not any real person. An ethereal silver female voice, warm and sweet, now carrying a private secret. Low-volume intimate delivery, slightly husky lower register, unhurried, expressive but restrained, no imitation.',
    }
    model=Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign',device_map='cpu',dtype=torch.float32,attn_implementation='sdpa')
    for index,(name,instruct) in enumerate(designs.items()):
      torch.manual_seed(20260926 + index)
      with torch.inference_mode():
        audio,sr=model.generate_voice_design(text=text,language='English',instruct=instruct,non_streaming_mode=True,max_new_tokens=2048)
      sf.write(out/f'{name}.wav',audio[0],sr)
      (out/f'{name}.txt').write_text(text,encoding='utf-8')
      print('✅ דגימת מקור',name,round(len(audio[0])/sr,2),'שניות',flush=True)
      del audio;gc.collect()


def mysterious_hebrew():
    from faster_whisper import WhisperModel
    from difflib import SequenceMatcher
    import subprocess,hashlib
    root=Path('mysterious-output');root.mkdir(exist_ok=True)
    target='יש לי סוד קטן, ואני אגלה לך אותו רק בסוף השיעור.'
    ipa=Path('mysterious-output/prompt.ipa').read_text(encoding='utf-8').strip()
    print('🔤 הגייה מנוקדת:',ipa,flush=True)
    model=WhisperModel('small',device='cpu',compute_type='int8')
    rows=[];fingerprints=set()
    for name in ('moon',):
      ref=Path('mysterious-references')/f'{name}.wav'
      short=root/f'{name}-short.wav'
      subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(ref),'-t','2.7','-af','afade=t=out:st=2.45:d=0.25',str(short)],check=True)
      out=root/f'{name}.wav'
      cmd=['qwen-tts','--model','/opt/models/talker.gguf','--codec','/opt/models/codec.gguf','--ref-wav',str(short),'--temp','0.55','--sub-temp','0.55','--seed','42','--lang','auto','-o',str(out)]
      print('⏳ יוצרת משפט מסתורי',name,flush=True)
      subprocess.run(cmd,input=ipa,text=True,check=True,stderr=subprocess.PIPE,timeout=240)
      audio,sr=sf.read(out)
      if not 2<len(audio)/sr<30: raise RuntimeError(f'Unexpected length for {name}')
      pcm=subprocess.check_output(['ffmpeg','-v','error','-i',str(out),'-f','s16le','-acodec','pcm_s16le','-'])
      digest=hashlib.sha256(pcm).hexdigest()
      if digest in fingerprints: raise RuntimeError(f'Duplicate PCM: {name}')
      fingerprints.add(digest)
      segments,_=model.transcribe(str(out),language='he',beam_size=5)
      heard=' '.join(seg.text.strip() for seg in segments)
      score=SequenceMatcher(None,target.replace(' ','').replace('.','').replace(',',''),heard.replace(' ','').replace('.','').replace(',','')).ratio()
      subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(out),'-c:a','libopus','-b:a','32k',str(root/f'{name}.ogg')],check=True)
      rows.append((name,round(score,3),heard,round(len(audio)/sr,2),digest))
      print('✅',name,heard,round(score,3),flush=True)
    lines=['| גרסה | התאמת תמלול (אינה מודדת מסתורין או מבטא) | תמלול | שניות | SHA-256 PCM |','|---|---:|---|---:|---|']
    lines += [f'| {n} | {sc:.3f} | {h} | {dur} | {sha[:12]} |' for n,sc,h,dur,sha in rows]
    (root/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('task',choices=['references','check-hebrew','compare-variants','mysterious-references','mysterious-hebrew'],nargs='?',default='references')
    args=parser.parse_args()
    {'references': references, 'check-hebrew': check_hebrew, 'compare-variants': compare_variants, 'mysterious-references': mysterious_references, 'mysterious-hebrew': mysterious_hebrew}[args.task]()

