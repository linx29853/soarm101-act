"""Export real-time dual-view clips from a completed 200_fixed evaluation."""
from pathlib import Path
import av
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
RUN='eval_200_fixed_temporal_20261109_123819'
OUT=ROOT/'docs/results/assets/200_fixed'

def frames(camera,start,end):
    p=ROOT/f'outputs/eval/{RUN}/videos/chunk-000/observation.images.{camera}/episode_000000.mp4'
    with av.open(str(p)) as c:
        st=c.streams.video[0];c.seek(int(start/float(st.time_base)),stream=st)
        target=start
        for f in c.decode(st):
            if f.time is None or float(f.time)<target:continue
            if target>=end:break
            yield target,f.to_image().resize((320,240),Image.Resampling.LANCZOS)
            target+=.1

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name,start,end in [('pick_place_01',0,8.5),('pick_place_02',20,32.5)]:
        seq=[]
        for (t,front),(_,hand) in zip(frames('front',start,end),frames('handeye',start,end)):
            canvas=Image.new('RGB',(640,265),'#17202a');canvas.paste(front,(0,25));canvas.paste(hand,(320,25))
            d=ImageDraw.Draw(canvas);d.text((8,6),f'Front | source {t:.1f}s | 1x',fill='white');d.text((328,6),'Handeye | 1x',fill='white');seq.append(canvas)
        seq[0].save(OUT/f'{name}.gif',save_all=True,append_images=seq[1:],duration=100,loop=0,optimize=False)
        seq[len(seq)//2].save(OUT/f'{name}_preview.jpg')
        print(name,len(seq),'frames',flush=True)
if __name__=='__main__':main()
