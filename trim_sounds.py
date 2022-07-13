# removes sound files that are not used in story segments (by adding to .gitignore)
# converts all sounds to mp3 to save space.
from pathlib import Path
import json,time,requests,pydub
DELETE_UNUSED_SOUNDS = True
RECONVERT_ALL_FILES = False # convert all to mp3 even if already exists WARNING: TAKES FOREVER
dirpath = r'.\assets\torappu\dynamicassets\audio'
varsFile = Path('story_variables.json')
if not varsFile.exists() or time.time() - varsFile.stat().st_mtime > 60*60*24:
    with requests.get('https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData/master/zh_CN/gamedata/story/story_variables.json') as r:
        if r.status_code == 200:
            with varsFile.open('wb') as f:
                f.write(r.content)
with varsFile.open('rb') as f:
    vars = json.load(f)
usedFiles = set()
for v in vars.values():
    try:
        if (v.lower().startswith('sound_beta_2')):
            usedFiles.add(v.lower())
    except:
        pass
with open('.gitignore','w') as gi:
    gi.writelines(['*\n','!*/\n'])
    for p in Path(dirpath).glob('**/*.*'):
        fname = str(p)[len(dirpath)-1:].lower()[:-4].replace('\\','/')
        if fname in usedFiles:
            print(p)
            abspath = p.resolve()
            gi.write('!'+str(p.with_suffix('.mp3')).replace("\\","/")+'\n')
            if RECONVERT_ALL_FILES or not abspath.with_suffix('.mp3').exists():
                sound = pydub.AudioSegment.from_wav(abspath)
                sound.export(abspath.with_suffix('.mp3'), format="mp3")
        elif DELETE_UNUSED_SOUNDS:
            p.resolve().unlink()
            