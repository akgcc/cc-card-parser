
import cv2
from pathlib import Path
import random
import numpy as np
import concurrent.futures
import json,sys
import collections.abc
import shutil
import copyreg
import pickle
import itertools
import statistics
import copy
import time
import requests
import re
from PIL import Image
import imagehash
from soul import calculate_soul
def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)

copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

MANUAL_DUPE_VERIFICATION = True # set to true if you can be present to verify dupes, press "y" if the displayed doctors are the same; press any other key otherwise.

# H S V
HIST_SIZE = [20,30,4]
HIST_RANGES = [0, 180] + [0, 256]+ [0, 256]
HIST_CHANNELS = list(range(len(HIST_SIZE)))

MAX_BAR_WIDTH = .065#% of image width; widest ive seen was 6.1%
MAX_TITLEBAR_HEIGHT = .05# % of image height
OP_ASPECT_RATIO = 130/156
SKILL_ICON_SIZE = 34/720 # at 720p
ELITE_ICON_SIZE = (25/720, 24/720) # at 720p # not a square its actually 25/24
TAG = sys.argv[1]
SQUADS = {}
OUTPUT_IMG_QUALITY = 80
OUTPUT_IMG_TYPE = '.jpg' # need to replace instances of IMWRITE_JPEG_QUALITY if you change this
JSON_DIR = Path('./json/').resolve()
SQUAD_JSON = JSON_DIR.joinpath(f'squads{TAG}.json')
DATA_JSON = JSON_DIR.joinpath(f'data{TAG}.json')
DATA_FIXES_JSON = JSON_DIR.joinpath(f'data{TAG}-fixes.json')
DATA_DUPE_FIXES_JSON = JSON_DIR.joinpath(f'data{TAG}-dupes.json')
doctorDir=Path(f'./doctors{TAG}/').resolve()
shutil.rmtree(doctorDir,True) # delete entire doctor dir to remove residual images, otherwise dupes which you moved to invalid will break the parser.
cropDir=Path(f'./cropped{TAG}/').resolve()
riskDir=Path(f'./risks{TAG}/').resolve()
numsDir=Path(f'./numberTemplates/').resolve()
imagesDir=Path(f'./images{TAG}/').resolve()
thumbsDir=Path(f'./thumbs{TAG}/').resolve()

avatarDir=Path('./avatars/').resolve()
skillDir=Path('./skills/').resolve()
eliteDir=Path('./elite/').resolve()
blankTemplate = Path('./BLANK_720.png').resolve()
charDataPath = Path('./character_table.json').resolve()
crisisDataPath = Path('./crisis_table.json').resolve()
skillDataPath = Path('./skill_table.json').resolve()
charDataPatchPath = Path('./char_patch_table.json').resolve()
failedParses = Path('./FAILED_PARSES.txt').resolve()
assertTests = Path('./assert_tests.p').resolve()
CC_START_DATES = {
'-ccbclear':1591995600#Friday, June 12, 2020 9:00:00 PM GMT
}

DATA_SOURCE = "https://raw.githubusercontent.com/Aceship/AN-EN-Tags/master/json/gamedata/"
CC_DATA_SOURCE = 'https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData/master/'
# DATA_SOURCE = 'https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData/master/'
# CC_DATA_SOURCE = DATA_SOURCE
# update character_table.json, and patch with char_patch_table
def update_char_table():
    if not crisisDataPath.exists() or time.time() - crisisDataPath.stat().st_mtime > 60*60*24:
        with requests.get(CC_DATA_SOURCE+'en_US/gamedata/excel/crisis_table.json') as r:
            if r.status_code == 200:
                with crisisDataPath.open('wb') as f:
                    f.write(r.content)
    if not charDataPath.exists() or time.time() - charDataPath.stat().st_mtime > 60*60*24:
        with requests.get(DATA_SOURCE+'en_US/gamedata/excel/character_table.json') as r:
            if r.status_code == 200:
                with charDataPath.open('wb') as f:
                    f.write(r.content)
    if not charDataPatchPath.exists() or time.time() - charDataPatchPath.stat().st_mtime > 60*60*24:
        with requests.get(DATA_SOURCE+'en_US/gamedata/excel/char_patch_table.json') as r:
            if r.status_code == 200:
                with charDataPatchPath.open('wb') as f:
                    f.write(r.content)
    if not skillDataPath.exists() or time.time() - skillDataPath.stat().st_mtime > 60*60*24:
        with requests.get(DATA_SOURCE+'en_US/gamedata/excel/skill_table.json') as r:
            if r.status_code == 200:
                with skillDataPath.open('wb') as f:
                    f.write(r.content)

    with charDataPath.open('rb') as f:
        data = json.load(f)
    with charDataPatchPath.open('rb') as f:
        datapatch = json.load(f)
    dictupdate(data, datapatch['patchChars'])
    # change guardmiya name
    data['char_1001_amiya2']['name'] = 'Guardmiya'
    with charDataPath.open('w') as f:
        json.dump(data, f)
        

    with crisisDataPath.open('rb') as f:
        crisis_table = json.load(f)
    for dat in crisis_table['seasonInfo']:
        num = dat['seasonId'].split('_')[2]
        CC_START_DATES[f'-cc{num}clear'] = int(dat['startTs'])
    
def dictupdate(d, u):
    if not isinstance(d, collections.abc.Mapping):
        return u
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dictupdate(d.get(k, {}), v)
        else:
            d[k] = v
    return d
def add_extra_data(data):
    for k,v in data.items():
        try:
            v['avgRarity'] = round(sum([CHAR_DATA[c['name']]['rarity'] for c in v['squad']])/len(v['squad'])+1,3)
            v['opcount'] = len(v['squad'])
        except Exception as e:
            print('ERROR ADDING EXTRA DATA',k,v,e)
    data = bubble_duplicates(data)
    return data
def bubble_duplicates(data):
    for k,v in data.items():
        if 'duplicate_of' not in v:
            continue
        while v['duplicate_of'] in data and 'duplicate_of' in data[v['duplicate_of']]:
            v['duplicate_of'] = data[v['duplicate_of']]['duplicate_of']
    return data
def group_duplicates(data):
    # instead of using "duplicate_of", group all dupes into "dupe_group"s
    # just use this inefficient method:
    groups = []
    for k,v in data.items():
        if 'duplicate_of' not in v:
            continue
        for g in groups:
            if v['duplicate_of'] in g:
                g.append(k)
                break
        else:
            groups.append([k,v['duplicate_of']])
    # add to data, but also remove all duplicate_of to save space.
    group_number = 0
    for g in groups:
        for k in g:
            data[k]['dupe_group'] = f'{group_number}{TAG}'
        group_number+=1
    for v in data.values():
        v.pop('duplicate_of',None)
    return data
def clean_output_folders(data):
    for dir in (cropDir,doctorDir,thumbsDir,riskDir):
        for file in dir.glob('*.*'):
            if file.with_suffix(OUTPUT_IMG_TYPE).name not in data:
                file.unlink()
def generate_data_json():
    with SQUAD_JSON.open('r') as f:
        data = json.load(f)
    for k,v in data.items():
        data[k] = {'squad': [{'name': '_'.join(i[0].split('_')[:3]).split('.')[0], 'skill': i[1], 'elite': i[2]} for i in v[0]],'group':clear_group(k),'support':{'name': '_'.join(v[1][0].split('_')[:3]).split('.')[0], 'skill': v[1][1], 'elite': v[1][2]} if v[1] else {'name':None}}
    for k in list(data.keys()):
        if not cropDir.joinpath(k).exists() and not cropDir.joinpath('duplicates/').joinpath(k).exists():
            del data[k]
    with DATA_JSON.open('w') as f:
        json.dump(data,f)
def fix_json_data(data):
    if DATA_FIXES_JSON.exists():
        with DATA_FIXES_JSON.open('r') as f:
            dictupdate(data,json.load(f))
    return data

def crop_sidebars(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,30,150,apertureSize = 3) # was 50,150
    h,w = im.shape[:2]
    # edges = cv2.Canny(gray,100,100,apertureSize = 3)
    minLineLength = .45 * h # .63 worked for most
    maxLineGap = 20/720 * h # was 100/720
    maxLineGap = 15/720 * h # was 100/720
    MIN_BAR_X = 10
    if DEBUG:
        cv2.imshow('sidebar_edges',edges)
    lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = minLineLength,maxLineGap = maxLineGap)
    if lines is None:
        return im
    x = 0
    if DEBUG:
        d = im.copy()
    # crop sides individually?
    left_x = 0
    right_x = w
    #left
    for line in [l for l in lines if abs(l[0][2]-l[0][0])<2 and l[0][0]>=MIN_BAR_X and l[0][0] <= w-MIN_BAR_X and (l[0][0] < MAX_BAR_WIDTH*w)][:1]:
        for x1,y1,x2,y2 in line:
            left_x = x1 + int(w * .01)
    #right
    for line in [l for l in lines if abs(l[0][2]-l[0][0])<2 and l[0][0]>=MIN_BAR_X and l[0][0] <= w-MIN_BAR_X and (l[0][0] > w-MAX_BAR_WIDTH*w)][:1]:
        for x1,y1,x2,y2 in line:
            right_x = x1 - int(w * .01)
    # if left bar was found, assume right bar is same width.
    if left_x:
        right_x = w - left_x
    im = im[:, left_x:right_x]
    return im

def crop_titlebar(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    h,w,_ = im.shape
    cv2.floodFill(gray,None,(w-5,5),255,0,0)
    cv2.floodFill(gray,None,(w//2,5),255,0,0)
    _,th = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)
    if DEBUG:
        cv2.imshow('titlebar_floodfill',th)
    edges = cv2.Canny(th,20,150,apertureSize = 5)
    
    wid_ratio = .65
    minLineLength = int(w*wid_ratio)
    maxLineGap = int(h * .1)
    MIN_TITLE_HEIGHT = 10
    if DEBUG:
        d = im.copy()
    lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = minLineLength,maxLineGap = maxLineGap)

    if lines is None:
        'titlebar not detected'
        return im

    y=0
    for line in [l for l in lines if abs(l[0][3]-l[0][1])<2 and l[0][1]>=MIN_TITLE_HEIGHT and l[0][1]<MAX_TITLEBAR_HEIGHT*h][:5]:
    # for line in lines:
        for x1,y1,x2,y2 in line:
            y = y2
            x = x1
            if DEBUG:
                cv2.line(d,(x1,y1),(x2,y2),(0,255,0),2)
    if DEBUG:
        cv2.imshow('titleline',d)
        cv2.waitKey()
    if y:
        y+= 1
        return im[y:h, :]
    return im
def parse_risks(data):

    testimg = 'asdf'#'1599804643036.jpg'#'1599766639097.jpg'#'1605112361906.jpg'#'1605544770468.png'#'1605204799361.png'#'1605132302340.png'#'1605156221752.png'
    # expects 2 digits, doesn't work on occluded numbers.
    RISK_NUMBER_TEMPLATES = [cv2.cvtColor(cv2.imread(str(numsDir.joinpath(f'{i}.png'))),cv2.COLOR_BGR2GRAY) for i in range(10)]
    LARGE_RISK_NUMBER_TEMPLATES = [cv2.resize(im,(int(im.shape[1]*1.15),int(im.shape[0]*1.15))) for im in RISK_NUMBER_TEMPLATES]
    # RISK_NUMBER_TEMPLATES = [cv2.Canny(image=cv2.GaussianBlur(im, (3,3), 0), threshold1=100, threshold2=200) for im in RISK_NUMBER_TEMPLATES]
    # LARGE_RISK_NUMBER_TEMPLATES = [cv2.Canny(image=cv2.GaussianBlur(im, (3,3), 0), threshold1=100, threshold2=200) for im in LARGE_RISK_NUMBER_TEMPLATES]
    riskpaths = list(riskDir.resolve().glob('*.*'))[:]
    for path in riskpaths:
        # print(path.name)
        im = cv2.imread(str(path))
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        lower_gray = np.array([0,0,75])
        upper_gray = np.array([255,15,255])#[255,10,255]
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if path.with_suffix(OUTPUT_IMG_TYPE).name in data:
                data[path.with_suffix(OUTPUT_IMG_TYPE).name]['risk'] = 0
            continue
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        
        # crop = mask[y:y+h,:]
        # oh,ow = crop.shape
        # crop=cv2.resize(crop, (int(ow*45/oh),45), interpolation = cv2.INTER_AREA)
        if path.name == testimg:
            # edges = cv2.Canny(image=crop, threshold1=100, threshold2=200)
            cv2.imshow('i',mask)
            cv2.waitKey()
             
        # mask = cv2.Canny(image=cv2.GaussianBlur(mask, (3,3), 0), threshold1=100, threshold2=200)
        digits = []
        
        # digits in jp client are scaled by ~115%
        # if first match in (1,2,3) we can assume?? its the left digit (we can't)
        baseline = None
        template_set = (0,1) # 0 is normal template, 1 is large
        for attempt in range(2):
            results = []
            for i in range(10):
                if 0 in template_set:
                    template = RISK_NUMBER_TEMPLATES[i]
                    res = cv2.matchTemplate(mask,template,cv2.TM_CCOEFF_NORMED)#,mask=template)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    results.append((max_val,i,max_loc,0))
                if 1 in template_set:
                    template = LARGE_RISK_NUMBER_TEMPLATES[i]
                    res = cv2.matchTemplate(mask,template,cv2.TM_CCOEFF_NORMED)#,mask=template)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    results.append((max_val,i,max_loc,1))
            if path.name == testimg:
                print(sorted(results))
            if 0 or baseline != None:
                # modify results by closeness to previous result baseline since numbers should be the same height
                results = [(a - abs(baseline-c[1])/100,b,c,d) for a,b,c,d in results]
            if path.name == testimg:
                print(sorted(results))
            score,num,loc,ts = sorted(results)[-1]
            if path.name == testimg:
                print(loc)
            if baseline == None:
                baseline = loc[1]
            # if len(template_set) > 1:
                # template_set = (ts,)
            h,w=RISK_NUMBER_TEMPLATES[i].shape
            mask[loc[1]:loc[1]+h, loc[0]:loc[0]+w]=0
            if path.name == testimg:
                cv2.imshow('i',mask)
                cv2.waitKey()
            if score > .5:#.25: #was .5 before edge # was .2
                digits.append((loc[0],num, score))
        loc_x_diff = 0
        if len(digits)> 1:
            loc_x_diff = abs(digits[0][0] - digits[1][0])
        # print(loc_x_diff)
        risk = int('0'+''.join([str(i[1]) for i in sorted(digits)]))
        if loc_x_diff > 44:# or (risk < 18 or risk > 35):
            risk = int(sorted(digits, key=lambda x: x[2])[-1][1])
        if path.with_suffix(OUTPUT_IMG_TYPE).name in data:
            if path.name == testimg:
                print(risk)
            data[path.with_suffix(OUTPUT_IMG_TYPE).name]['risk'] = risk
    return data



# CC_START_DATES = {
	# '-ccbclear': 1592002800, #2020 june 12 16:00 UTC-7
	# '-cc0clear': 1599750000, #2020 sept 10 8:00 UTC-7
	# '-cc1clear': 1605114000, #2020 nov 11 10:00 UTC-7
	# '-cc2clear': 1612458000, #2021 feb 4 10:00 UTC-7
	# '-cc3clear': 1622221200, #2021 may 28 10:00 UTC-7
	# '-cc4clear': 1626195600, #2021 july 13 10:00 UTC-7
    # '-cc5clear': 1636650000, #2021 nov 11 10:00 UTC-7
	# server reset and therefore week 2 is at 0400 UTC-7
# }

def clear_group(fname):
    # 0 == day1, 1 == week1 2==week2
    try:
        ts = int(fname.split('.')[0])//1000
    except:
        return 2
    if ts < (CCSTART + 172800) - (CCSTART % (60 * 60 * 24)) + 39600: # day "1" end
        return 0
    if ts < (CCSTART + 604800) - (CCSTART % (60 * 60 * 24)) + 39600: # week 1 end
        return 1
    return 2
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
def remove_duplicates(data):
    # now uses imagehash along with ssim
    global MANUAL_DUPE_VERIFICATION
    # move all files OUT of duplicates first
    for dir in (cropDir, thumbsDir):
        for file in dir.joinpath('duplicates/').glob('*.*'):
            try:
                file.rename(dir.joinpath(file.name))
            except FileExistsError:
                file.unlink()
    def archiveClear(file, originalFile, move_files):
        for dir in (cropDir, thumbsDir):
            if dir.joinpath(file.name).exists():
                data[file.name]['duplicate_of'] = originalFile.name
                # only move dupes that won't be displayed.
                if move_files:
                    try:
                        dir.joinpath(file.name).rename(dir.joinpath('duplicates/').joinpath(file.name))
                    except FileExistsError:
                        dir.joinpath(file.name).unlink()
        for dir in (doctorDir, ):
            oDir = dir.joinpath('duplicates/').joinpath(originalFile.stem)
            oDir.mkdir(exist_ok=True)
            if dir.joinpath(file.with_suffix('.png').name).exists():
                try:
                    shutil.copy(dir.joinpath(file.with_suffix('.png').name) ,oDir.joinpath(file.with_suffix('.png').name))
                    shutil.copy(dir.joinpath(originalFile.with_suffix('.png').name) ,oDir.joinpath(originalFile.with_suffix('.png').name))
                except FileExistsError:
                    pass
   
    # KP_TH = 70
    # KP_TH_LOW = 50
    # HIGH_TH = .6
    # MANUAL_VERIFY_TH = .5
    
    def images_are_similar(a,b):
        kp_1, desc_1 = sift_data[a[0]]
        kp_2, desc_2 = sift_data[b[0]]
        if len(kp_1)<2 or len(kp_2)<2:
            return 0,0
        matches = G_BF.knnMatch(desc_1,desc_2, k=2)
        good_points = []
        for m,n in matches:
            if m.distance <.7*n.distance: # lower ratio is stricter.
                good_points.append(m)
        number_keypoints = max(len(kp_1),len(kp_2))
        # if len(good_points)/number_keypoints > .65 and  '1623332632019.png' in (a[0].name,b[0].name):
            # print('similar with',len(good_points),number_keypoints,'kps',len(good_points)/number_keypoints,'to',b[0].name)
        return len(good_points)/number_keypoints,number_keypoints
        
    paths = list(doctorDir.glob('*.*'))
    def merge(d,v):
        for k in v:
            d.add(k)
            if k in groups:
                a = groups[k]
                del groups[k]
                merge(d,a)
        return d
    dupes_json = {}
    # load manually assigned dupes
    if DATA_DUPE_FIXES_JSON.exists():
        with DATA_DUPE_FIXES_JSON.open('r') as f:
            dupes_json = json.load(f)
    # MANUAL_DUPE_VERIFICATION = 0
    # remove invalid dupe pairs (original file was made invalid)
    
    groups = {}
    hashfunc = imagehash.phash
    
    # paths = [p for p in paths if p.stem in ('1655433607081','1654830094868')]
    # paths = [p for p in paths if p.stem in ('1655471405936','1654865775725')]
    # paths = [p for p in paths if p.stem in ('1655554223310','1655229086576')]
    # paths = [p for p in paths if p.stem in ('1654835079802','1654996932072')]
    cv_ims = {p : cv2.imread(str(p)) for p in paths}
    fast = cv2.FastFeatureDetector_create()
    ims_FAST = {k:len(fast.detect(v,None)) for k,v in cv_ims.items()}
    # print(ims_FAST.values())
    
    FAST_TH = 65 # any fewer and we throw out this image (mostly solid color)
    
    valid_paths = [p for p in paths if ims_FAST[p] > FAST_TH]
    # print('valid paths:', len(valid_paths),'/',len(paths))
    # print([ims_FAST[p] for p in paths ])
    im_hashes = [(p,hashfunc(Image.open(p))) for p in valid_paths]
    
    # use ssim to filter weak imagehash matches
    set1 = [(k,cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)) for k,v in cv_ims.items()]
    lower_gray = np.array([0,0,100]) # 95 works but leave some leeway
    upper_gray = np.array([255,255,255])
    hsv = [cv2.cvtColor(v, cv2.COLOR_BGR2HSV) for v in cv_ims.values()]
    grays = [cv2.inRange(cv2.blur(a,(7,7)), lower_gray, upper_gray) for a in hsv]
    edges = [cv2.Canny(a,50,150,apertureSize = 3) for a in grays] 
    contours = [cv2.findContours(ed,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[0] for ed in edges]
    cntminxs = [cv2.boundingRect(min(c, key = lambda x: cv2.boundingRect(x)[0]))[0] if c else 0 for c in contours]
    set1 = [(i[0],cv2.warpAffine(i[1], np.float32([[1,0,-cntminxs[idx]],[0,1,0]]), (i[1].shape[1], i[1].shape[0]))) for idx,i in enumerate(set1)]
    sift_data = {a[0]:G_SIFT.detectAndCompute(a[1], None) for a in set1}
    
    ssim_kp_th = 95 # very high value to avoid false positives
    ssim_score_th = [.5,.65] # lower th for manual verification, higher th is automatic (.6 might be ok but err on the safe side)
    dupe_th = 2
    possible_dupe_th = 4
    # stretch_dupe_th no longer used (effectively infinity)
    # stretch_dupe_th = 20 # this value is extremely high, but used to catch outliers (5 would be reasonable if not using ssim to check)
    for a,b in itertools.combinations(im_hashes,2):
        dupekey = '|'.join(sorted([a[0].name,b[0].name]))
        is_dupe = dupes_json.get(dupekey,False)
        # print(a[1]-b[1],images_are_similar(a,b))
        if a[1]-b[1] < dupe_th:
            is_dupe = True
        # elif a[1]-b[1] < possible_dupe_th:
            # if MANUAL_DUPE_VERIFICATION and dupekey not in dupes_json:
                # d = np.concatenate((cv_ims[a[0]], cv_ims[b[0]]), axis=1)
                # cv2.imshow('press y if these are the same, or another key if not',d)
                # print('ssim:',score,kps)
                # k = cv2.waitKey(15000)
                # if k==ord('y'):
                    # dupes_json[dupekey] = 1
                # elif k != -1:
                    # dupes_json[dupekey] = 0
                # else:
                    # print('Timed out, skipping manual dupe verification.')
                    # MANUAL_DUPE_VERIFICATION = False
                # cv2.destroyAllWindows()
        elif dupekey not in dupes_json:# and a[1]-b[1] < stretch_dupe_th:
            score,kps = images_are_similar(a,b)
            if score > ssim_score_th[1] and (kps > ssim_kp_th or score == 1): # these MUST be the same doctor (use a high th)
                # is a dupe
                is_dupe = True
                # if a[1]-b[1] > 4:
                    # d = np.concatenate((cv_ims[a[0]], cv_ims[b[0]]), axis=0)
                    # cv2.imshow('THESE BETTER BE THE SAME',d)
                    # print('ssim:',a[1]-b[1],score,kps)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
            elif MANUAL_DUPE_VERIFICATION and score > ssim_score_th[0]: #  and kps > ssim_kp_th:
                # potentially a dupe
                d = np.concatenate((cv_ims[a[0]], cv_ims[b[0]]), axis=0)
                cv2.imshow('press y if these are the same, or another key if not',d)
                k = cv2.waitKey(60000)
                if k==ord('y'):
                    dupes_json[dupekey] = 1
                    is_dupe = True
                elif k != -1:
                    dupes_json[dupekey] = 0
                else:
                    print('Timed out, skipping manual dupe verification.')
                    MANUAL_DUPE_VERIFICATION = False
                time.sleep(.25)
                print('ssim:',a[1]-b[1],score,kps,is_dupe,a[0].name,b[0].name)
                cv2.destroyAllWindows()
        if is_dupe:
            groups.setdefault(a[0],[]).append(b[0])
            groups.setdefault(b[0],[]).append(a[0])
    with DATA_DUPE_FIXES_JSON.open('w') as f:
        json.dump(dupes_json, f)
        
    grouped = []
    for k in list(groups.keys()):
        if k in groups:
            a = groups[k]
            del groups[k]
            grouped.append(merge(set([k]),a))
    print('TOTAL DUPES FOUND:',sum([len(v) for v in grouped]))
    for i,g in enumerate(grouped):
        dupes = sorted(g,key=lambda x: x.stem)# sort by filename, this makes a big assumption that all filenames are timestamps.
        groups = set([data[dupes[-1].with_suffix(OUTPUT_IMG_TYPE).name]['group']])
        # res = np.concatenate([cv_ims[p] for p in dupes], axis=1)
        # cv2.imshow('dupegroup', res)
        # cv2.waitKey(0)
        for d in dupes[-2::-1]:# reversed but skip first (prev last) element
            group = data[d.with_suffix(OUTPUT_IMG_TYPE).name]['group']
            archiveClear(d.with_suffix(OUTPUT_IMG_TYPE), dupes[-1].with_suffix(OUTPUT_IMG_TYPE), group in groups)
            groups.add(group)
def remove_dupes_ssim(data):
    # old dupe finder, before switching to phash
    global MANUAL_DUPE_VERIFICATION
    # move all files OUT of duplicates first
    for dir in (cropDir, thumbsDir):
        for file in dir.joinpath('duplicates/').glob('*.*'):
            try:
                file.rename(dir.joinpath(file.name))
            except FileExistsError:
                file.unlink()
    def archiveClear(file, originalFile, move_files):
        for dir in (cropDir, thumbsDir):
            if dir.joinpath(file.name).exists():
                data[file.name]['duplicate_of'] = originalFile.name
                # only move dupes that won't be displayed.
                if move_files:
                    try:
                        dir.joinpath(file.name).rename(dir.joinpath('duplicates/').joinpath(file.name))
                    except FileExistsError:
                        dir.joinpath(file.name).unlink()
        for dir in (doctorDir, ):
            oDir = dir.joinpath('duplicates/').joinpath(originalFile.stem)
            oDir.mkdir(exist_ok=True)
            if dir.joinpath(file.with_suffix('.png').name).exists():
                try:
                    shutil.copy(dir.joinpath(file.with_suffix('.png').name) ,oDir.joinpath(file.with_suffix('.png').name))
                    shutil.copy(dir.joinpath(originalFile.with_suffix('.png').name) ,oDir.joinpath(originalFile.with_suffix('.png').name))
                except FileExistsError:
                    pass
    paths = list(doctorDir.glob('*.*'))
    def merge(d,v):
        for k in v:
            d.add(k)
            if k in groups:
                a = groups[k]
                del groups[k]
                merge(d,a)
        return d
    sift_data = {}
    
    KP_TH = 70
    KP_TH_LOW = 50
    def images_are_similar(a,b):
        kp_1, desc_1 = sift_data[a[0]]
        kp_2, desc_2 = sift_data[b[0]]
        if len(kp_1)<2 or len(kp_2)<2:
            return 0,0
        matches = G_BF.knnMatch(desc_1,desc_2, k=2)
        good_points = []
        for m,n in matches:
            if m.distance <.7*n.distance: # lower ratio is stricter.
                good_points.append(m)
        number_keypoints = max(len(kp_1),len(kp_2))
        # if len(good_points)/number_keypoints > .65 and  '1623332632019.png' in (a[0].name,b[0].name):
            # print('similar with',len(good_points),number_keypoints,'kps',len(good_points)/number_keypoints,'to',b[0].name)
        return len(good_points)/number_keypoints,number_keypoints

    im = [(p,cv2.imread(str(p))) for p in paths]
    imsize = im[0][1].shape[0]*im[0][1].shape[1]
    # hist_data = {a[0]:cv2.calcHist(a[1], list(range(3)), None, [50,50,50], [0,256]+[0,256]+[0,256], accumulate=False) for a in im}
    
    set1 = [(i[0],cv2.cvtColor(i[1], cv2.COLOR_BGR2GRAY)) for i in im]
    # edges = [cv2.Canny(a[1],50,150,apertureSize = 3) for a in set1] 
    # contours = [cv2.findContours(ed,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[0] for ed in edges]
    # cntareas = [sum(map(cv2.contourArea,cnts)) for cnts in contours]
    # cntpercents = [c/imsize for c in cntareas]
    # set1 = [v for i,v in enumerate(set1) if cntpercents[i]>.08 and len(contours[i])>3]
    # set1 = [v for i,v in enumerate(set1) if cntpercents[i]>.06 and len(contours[i])>3]
    # set1 = [v for i,v in enumerate(set1) if len(contours[i])>3]
    

    lower_gray = np.array([0,0,100]) # 95 works but leave some leeway
    upper_gray = np.array([255,255,255])
    hsv = [cv2.cvtColor(i[1], cv2.COLOR_BGR2HSV) for i in im]
    grays = [cv2.inRange(cv2.blur(a,(7,7)), lower_gray, upper_gray) for a in hsv]
    edges = [cv2.Canny(a,50,150,apertureSize = 3) for a in grays] 
    contours = [cv2.findContours(ed,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[0] for ed in edges]
    cntminxs = [cv2.boundingRect(min(c, key = lambda x: cv2.boundingRect(x)[0]))[0] if c else 0 for c in contours]
    set1 = [(i[0],cv2.warpAffine(i[1], np.float32([[1,0,-cntminxs[idx]],[0,1,0]]), (i[1].shape[1], i[1].shape[0]))) for idx,i in enumerate(set1)]
    
    # d = im[0][1].copy()
    
    # cv2.imshow('c',transformedimages[0][1])
    # cv2.waitKey()
    # print('found',len(contours[0]),'contours')
    # c = min(contours[0], key = lambda x: cv2.boundingRect(x)[0])
    # # for c in contours[0]:
    # x,y,w,h = cv2.boundingRect(c)
    # cv2.rectangle(d,(x,y),(x+w,y+h),(200,200,200),2)
    
    # #transform image to line up with contour
    # matrix = [[1,0,-x],
              # [0,1,0]]
    # t = np.float32(matrix)
    # h, w = im[0][1].shape[:2]
    # d = cv2.warpAffine(d, t, (w, h))
    
    # cv2.imshow('c',d)
    # cv2.waitKey()
    # exit()
    
    # filter set of images then gen sift data
    # sift_data = {a[0]:G_SIFT.detectAndCompute(unsharp_mask(a[1]), None) for a in set1}
    sift_data = {a[0]:G_SIFT.detectAndCompute(a[1], None) for a in set1}
    
    # dupes = [a for a in set1 if a[0].name in ('1654835057555.png','1654983227637.png')]
    # dupes = [a for a in set1 if a[0].name in ('1626298768237.png','1627344552523.png')]
    # dupes = [a for a in set1 if a[0].name in ('1626413456716.png','1626776973041.png')]
    # dupes = [a for a in set1 if a[0].name in ('1626277927333.png','1626754282586.png')]
    # dupes = [a for i,a in enumerate(set1) if a[0].name in ('1627193775955.jpg','1626783169245.png')]
    # dupes = [a for i,a in enumerate(set1) if a[0].name in ('1626427723017.png','1627193775955.jpg')]
    # dupes = [a for i,a in enumerate(set1) if a[0].name in ('1627006712830.png','1626450296491.jpg')]
    
    # cv2.imshow('1',dupes[0][1])
    # cv2.imshow('2',dupes[1][1])
    # print(images_are_similar(dupes[0],dupes[1]),dupes[0][0].name,dupes[1][0].name)
    # cv2.waitKey()
    # exit()
    # for i in dupes:
        # cv2.imshow('db',i[1])
        # cv2.waitKey()
    # exit()
    

    groups = {}
    sim_map = {}
    HIGH_TH = .6
    MANUAL_VERIFY_TH = .5
    # LOW_TH = .6
    
    
    dupes_json = {}
    # load manually assigned dupes
    if DATA_DUPE_FIXES_JSON.exists():
        with DATA_DUPE_FIXES_JSON.open('r') as f:
            dupes_json = json.load(f)
            sim_map = copy.deepcopy(dupes_json)
            # for k,v in dupes_json.items():
                # if 'duplicate_of' in v:
                    # sim_map[k+v['duplicate_of']] = (2,KP_TH+1)
                    # sim_map[v['duplicate_of']+k] = (2,KP_TH+1)
                    # sim_map[k] = v
    # print(sim_map[dupes[0][0].name+dupes[1][0].name])
    # exit()
    # inefficient but luckily not too slow since images are small.
    for a,b in itertools.combinations(set1,2):
        if a[0].name+b[0].name not in sim_map:
            score,kpcount = images_are_similar(a,b)
            sim_map[a[0].name+b[0].name] = (score,kpcount)
            # if score > HIGH_TH and kpcount > KP_TH_LOW:
                # groups.setdefault(a[0],[]).append(b[0])
                # groups.setdefault(b[0],[]).append(a[0])
            if MANUAL_DUPE_VERIFICATION and score > MANUAL_VERIFY_TH and kpcount > KP_TH_LOW and (score <= HIGH_TH or kpcount <= KP_TH_LOW):
                d = np.concatenate((a[1],b[1]), axis=1)
                cv2.imshow('press y if these are the same, or another key if not',d)
                k = cv2.waitKey(15000)
                if k==ord('y'):
                    sim_map[a[0].name+b[0].name] = (2,999)#KP_TH+1)
                    dupes_json[a[0].name+b[0].name] = (2,999)#KP_TH+1)
                    dupes_json[b[0].name+a[0].name] = (2,999)
                    # v = dupes_json.setdefault(min((a[0].name,b[0].name)),{})
                    # v['duplicate_of'] = max((a[0].name,b[0].name))
                    print('LINKING AS DUPE')
                    groups.setdefault(a[0],[]).append(b[0])
                    groups.setdefault(b[0],[]).append(a[0])
                elif k != -1:
                    sim_map[a[0].name+b[0].name] = (0,0)#KP_TH_LOW-1)
                    dupes_json[a[0].name+b[0].name] = (0,0)#KP_TH_LOW-1)
                    dupes_json[b[0].name+a[0].name] = (0,0)
                else:
                    print('Timed out, skipping manual dupe verification.')
                    MANUAL_DUPE_VERIFICATION = False
                cv2.destroyAllWindows()
        score,kpcount = sim_map[a[0].name+b[0].name]
        if score > HIGH_TH and kpcount > KP_TH_LOW:
            groups.setdefault(a[0],[]).append(b[0])
            groups.setdefault(b[0],[]).append(a[0])
    with DATA_DUPE_FIXES_JSON.open('w') as f:
        json.dump(dupes_json, f)
    grouped = []
    for k in list(groups.keys()):
        if k in groups:
            a = groups[k]
            del groups[k]
            grouped.append(merge(set([k]),a))
    print('TOTAL DUPES FOUND:',sum([len(v) for v in grouped]))
    for i,g in enumerate(grouped):
        dupes = sorted(g,key=lambda x: x.stem)# sort by filename, this makes a big assumption that all filenames are timestamps.
        com = [sim_map.get(a.name+b.name, sim_map.get(b.name+a.name,[0]))[0]>HIGH_TH for a,b in itertools.combinations(dupes,2)]
        if not (all(com)) and not [x>1 for x in com]:# or dupes[-1].name == '1626754282586.png':
            sums = {}
            scores = {}
            kpcount = []
            for a,b in itertools.combinations(dupes,2):
                sc,kps = sim_map.get(a.name+b.name, sim_map.get(b.name+a.name,[0,0]))
                kpcount.append(kps)
                sums.setdefault(a.name,[]).append(sc)
                sums.setdefault(b.name,[]).append(sc)
            for k,v in sums.items():
                scores[k] = sum(v)/len(v)
                # try using median isntead of mean score.
                # scores[k] = statistics.median(v)
                # print(v,scores[k])
            
            avg_kps = sum(kpcount)/len(kpcount)
            # print('avsc',sum(scores.values())/len(scores),sum(scores.values())/len(scores)-min(scores.values()))
            # print('med',statistics.median(scores.values()),max(scores.values())-min(scores.values()))
            # print('sd',statistics.stdev(scores.values()),statistics.harmonic_mean(scores.values()))
            if avg_kps < KP_TH:
                # trash this entire set, its probably blacked out names
                # print('throwing out due to low kp avg',avg_kps,dupes[-1].name)
                continue
            # med_mod = statistics.median(scores.values()) - (max(scores.values())-min(scores.values()))/2
            mean_mod = statistics.harmonic_mean(scores.values()) - statistics.stdev(scores.values())*1.5
            # filter out any with avg score below LOW_TH
            dupes = [d for d in dupes if scores[d.name] >= mean_mod]
            # print('not all match for ',dupes[-1].name,'filtering those below',mean_mod)
            # print(scores)
        # print([(d.name,data[d.name]['group']) for d in dupes])
        groups = set([data[dupes[-1].with_suffix(OUTPUT_IMG_TYPE).name]['group']])
        for d in dupes[-2::-1]:# reversed but skip first (prev last) element
            group = data[d.with_suffix(OUTPUT_IMG_TYPE).name]['group']
            archiveClear(d.with_suffix(OUTPUT_IMG_TYPE), dupes[-1].with_suffix(OUTPUT_IMG_TYPE), group in groups)
            groups.add(group)
def createThumbs():
    # for paths in (cropDir.glob('*.*'),cropDir.joinpath('duplicates/').glob('*.*')):
        for path in cropDir.glob('*.*'):
            im=cv2.imread(str(path))
            thumb=cv2.resize(im,(295,166), interpolation = cv2.INTER_AREA)
            cv2.imwrite(str(thumbsDir.joinpath(path.name)),thumb,[int(cv2.IMWRITE_JPEG_QUALITY), OUTPUT_IMG_QUALITY])
            # cv2.imwrite(str(thumbsDir.joinpath(path.name)),thumb,[int(cv2.IMWRITE_WEBP_QUALITY), OUTPUT_IMG_QUALITY])
def smoothimg(im):
    kernel = np.ones((5,5), np.uint8)
    d = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    # d = cv2.dilate(d, kernel, iterations=1)
    return d
def get_rightmost_blank(im_gray,template,force=False,ignore_rightmost=0):
    #destructive on im_gray, will blank out matches.
    forced = False
    # im_gray = im_gray_original.copy()
    def match(tmp):
        matches = []
        while 1:
            result = cv2.matchTemplate(im_gray, tmp, cv2.TM_CCOEFF_NORMED)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
            if maxVal <.54: # worked at .6 except images-cc0clear\1600883194132.png
                if DEBUG:
                    print('value too low',maxVal)
                break
            (startX, startY) = maxLoc
            endX = startX + tmp.shape[1]
            endY = startY + tmp.shape[0]
            matches.append((startX,(startX, startY, endX, endY)))
            im_gray[startY:endY, startX:endX]=0
        if not matches[ignore_rightmost:]:
            return None
        return sorted(matches, reverse=True)[ignore_rightmost:][0][1]
    right_coords = match(template)
    if force and not right_coords:
        # template couldn't match, possibly due to crop
        # try matching half template instead
        print('forcing match w/ half template...')
        forced = True
        half_template = template[:,0:int(template.shape[1]/2)]
        right_coords = match(half_template)
        if not right_coords:
            #try again with quarter template
            print('forcing match w/ quarter template...')
            quarter_template = template[0:int(template.shape[0]/2),0:int(template.shape[1]/2)]
            right_coords = match(quarter_template)
    return forced,right_coords
LOWER_GRAY = np.array([0,0,0])
UPPER_GRAY = np.array([255,45,255])
# OPMASK = np.zeros((180,180,1), np.uint8)
def avatar_hist(im, templates = True, use_mask = True, debug=False):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    histomask = cv2.bitwise_not(cv2.inRange(hsv, LOWER_GRAY, UPPER_GRAY))
    if debug :
        cv2.imshow('c',histomask)
        cv2.waitKey()
    if templates:
        mask = np.zeros(histomask.shape,np.uint8)
        mask[0:histomask.shape[1],0:histomask.shape[1]] = cv2.resize(OPMASK, (histomask.shape[1],histomask.shape[1]), interpolation = cv2.INTER_AREA)
        histomask = cv2.bitwise_or(mask,histomask)
    histomask = smoothimg(histomask)
    h,w = im.shape[:2]
    masked = cv2.bitwise_and(im[0:int(h*AVATAR_CROP),0:w],im[0:int(h*AVATAR_CROP),0:w],mask = histomask[0:int(h*AVATAR_CROP),0:w])
    if debug :
        cv2.imshow('o',im)
        cv2.imshow('c',histomask)
        cv2.waitKey()
    hist_base = cv2.calcHist([hsv[0:int(hsv.shape[0]*AVATAR_CROP),:]], HIST_CHANNELS, histomask[0:int(hsv.shape[0]*AVATAR_CROP),:] if use_mask else None, HIST_SIZE, HIST_RANGES, accumulate=False)
    cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist_base
def mostly_gray(im):
    # returns true if an image is mostly greyscale. used to find blank operator slots
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im_gs = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lower_gray = np.array([0,1,0]) # was 0,1,0 this may be unsafe
    upper_gray = np.array([255,255,255])

    gray = cv2.inRange(cv2.blur(im_hsv,(5,5)), lower_gray, upper_gray)
    gray_percent = 1-(np.count_nonzero(gray)/(gray.shape[0]*gray.shape[1]))
    
    GRAY_MIN_PERCENT = .3
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    template = cv2.cvtColor(cv2.imread(str(blankTemplate)), cv2.COLOR_BGR2GRAY)

    h,w = im.shape[:2]
    th,tw = template.shape[:2]

    result = cv2.matchTemplate(im_gs[0:th,0:tw], template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    if maxVal > .9:
        return True
    contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if not len(contours):
        return True
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    if DEBUG:
        print('gray_contours:',len(contours),abs(im.shape[0]-h),abs(im.shape[1]-w),im.shape[1],gray_percent,maxVal)
    return len(contours)==0 or (
    abs(im.shape[0]-h) <3 \
    and abs(im.shape[1]-w) > im.shape[1]*.12 \
    and gray_percent >  GRAY_MIN_PERCENT \
    and maxVal > .1)
    # if len(contours):
        # c = max(contours, key = cv2.contourArea)
        # x,y,w,h = cv2.boundingRect(c)
        # if abs(im.shape[0]-h) <2:
            # print('contour:',x,h,im.shape[0])
        # gray_area_percent = 1 - (cv2.contourArea(c) / (im.shape[1]*im.shape[0]))
    # else:
        # gray_area_percent = 1
    # d=im.copy()
    # print(len(contours),abs(im.shape[0]-h),abs(im.shape[1]-w),'CNTFOUND',gray_percent,gray_area_percent)
    # for cnt in contours:
        # x,y,w,h = cv2.boundingRect(cnt)
        # cv2.rectangle(d,(x,y),(x+w,y+h),(200,0,0),2)
    # cv2.imshow('th',gray)
    # cv2.imshow('cont',d)
    # cv2.waitKey()

def match_op(roi):
    if mostly_gray(roi):
        return 1,BLANK_NAME
    # cv2.imshow('roi',roi)
    # cv2.waitKey()
    def sift_diff(kp_1,desc_1,av_name):
        kp_2,desc_2 = av_sifts[av_name]
        if len(kp_1)<2 or len(kp_2)<2:
            return 0
        matches = G_BF.knnMatch(desc_1,desc_2, k=2)
        good_points = []
        for m,n in matches:
            if m.distance <.75*n.distance:
                good_points.append(m)
        number_keypoints = max(len(kp_1),len(kp_2))
        return len(good_points)/number_keypoints
    def all_sift_diffs(kp_1,desc_1):
        # higher value is better 0-1
        diffs = []
        for name,val in av_sifts.items():
            kp_2,desc_2 = val
            if len(kp_1)<2 or len(kp_2)<2:
                diffs.append((0,name))
                continue
            matches = G_BF.knnMatch(desc_1,desc_2, k=2)
            good_points = []
            for m,n in matches:
                if m.distance <.75*n.distance:
                    good_points.append(m)
            number_keypoints = max(len(kp_1),len(kp_2))
            percent_sim = len(good_points)/number_keypoints
            diffs.append((percent_sim,name))
        return diffs

    hist_base = avatar_hist(roi)
    matches = [(cv2.compareHist(hist_base, k, cv2.HISTCMP_BHATTACHARYYA),k,v,im) for k,v,im in av_hists]
    blank_match = cv2.compareHist(avatar_hist(roi,use_mask=False), BLANK_HIST, cv2.HISTCMP_BHATTACHARYYA)
    res = sorted(matches,key=lambda x:x[0],reverse=False)
    
    if res[0][2].name.endswith(BLANK_NAME):
        return res[0][0],res[0][2].name
    op = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kp_1,desc_1 = G_SIFT.detectAndCompute(op, None)
    
    hist_match_ok = sift_diff(kp_1,desc_1,res[0][2].name)
    if hist_match_ok > .1:
        return hist_match_ok,res[0][2].name
    options = all_sift_diffs(kp_1,desc_1)
    if DEBUG:
        print('match below th:', res[0][0],res[0][2].name)
        print('insted used',sorted(options,key=lambda x: x[0],reverse=True)[:5])
    best = sorted(options,key=lambda x: x[0],reverse=True)[0]
    if best[0] > .108: # .11 failed on this single image: -cc4clear/1626719849150.jpg, .105 fails in another way.
        return best
        
        
    
    ROI_EXPANDED = np.zeros((roi.shape[0]*2,roi.shape[1]*2, 4), np.uint8)
    for c in range(0, 3):
        ROI_EXPANDED[int(roi.shape[0]/2) : int(roi.shape[0]*1.5), int(roi.shape[1]/2) : int(roi.shape[1]*1.5),c] = roi[:,:,c]
    def template_compare(avatar_img_name):
        if avatar_img_name.endswith(BLANK_NAME):
            return -1
        orig_avatar = cv2.imread(str(avatarDir.joinpath(avatar_img_name)), cv2.IMREAD_UNCHANGED)
        # scale based on width alone
        rez = cv2.resize(orig_avatar, (roi.shape[1],roi.shape[1]), interpolation = cv2.INTER_AREA)
        res = cv2.matchTemplate(ROI_EXPANDED,rez,cv2.TM_CCOEFF_NORMED)#,mask=template)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return max_val
    
    # compare only against the best match, could pick the best match out of the list, but this isn't accurate.
    # if template_compare(best[1]) > template_compare(str(res[0][2])):
        # return best
  
    # check aspect ratio, if too thin assume blank
    if abs(roi.shape[1]/roi.shape[0] - OP_ASPECT_RATIO) > .3:
        return 1,BLANK_NAME
    return res[0][0],res[0][2].name
def match_skill(opimg, opname, full_img_height):
    ' return skill # 1-3 or 0 if no skill or invalid operator '
    opkey = '_'.join(opname.split('_')[:3]).split('.')[0]
    if opkey not in CHAR_DATA:
        return 0
    best,skill = 0,0
    for i,skname in enumerate([v['skillId'] for v in CHAR_DATA[opkey]['skills']]):
        # scale img based on height
        # s = int(opimg.shape[0]*34/156)
        s = int(SKILL_ICON_SIZE * full_img_height)
        template = cv2.resize(SKILL_ICONS[skname], (s,s), interpolation = cv2.INTER_AREA)
        res = cv2.matchTemplate(opimg,template,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > best:
            best = max_val
            skill = i+1
    if best < .45:
        # match again with middle of template.
        for i,skname in enumerate([v['skillId'] for v in CHAR_DATA[opkey]['skills']]):
            s = int(SKILL_ICON_SIZE * full_img_height)
            template = cv2.resize(SKILL_ICONS[skname], (s,s), interpolation = cv2.INTER_AREA)
            template = template[template.shape[0]*1//5:template.shape[0]*4//5, template.shape[0]*1//5:template.shape[0]*4//5]
            res = cv2.matchTemplate(opimg,template,cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > best:
                best = max_val
                skill = i+1
    return skill
def match_elite(opimg, opname, full_img_height):
    ' return elite level # 0-2 '
    opkey = '_'.join(opname.split('_')[:3]).split('.')[0]
    if opkey not in CHAR_DATA:
        return 0
    best,elite = 0,0
    for i in range(1,len(CHAR_DATA[opkey]['phases'])):
        # scale img based on width
        scale_factor = ELITE_ICON_SIZE[0] * full_img_height / ELITE_ICONS[str(i)].shape[1]
        # print(ELITE_ICONS[str(i)].shape[1], scale_factor)
        template = cv2.resize(ELITE_ICONS[str(i)], (int(scale_factor * ELITE_ICONS[str(i)].shape[1]),int(scale_factor * ELITE_ICONS[str(i)].shape[0])), interpolation = cv2.INTER_AREA)
        # cv2.imshow('sidebar_edges',template)
        # cv2.imshow('sideasdbar_edges',ELITE_ICONS['1'])
        # cv2.imshow('sidebar_edges',opimg[0:opimg.shape[0], 0:int(opimg.shape[1])])
        
        # cv2.waitKey(0)
        # exit()
        # print('SHAPES',ELITE_ICONS[str(i)].shape, template.shape)
        #crop off right 1/4 of op img so potentials won't match elite
        res = cv2.matchTemplate(opimg[0:opimg.shape[0], 0:int(opimg.shape[1]*1)],template,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # print(i, max_val,max_loc)
        if max_val > best:
            best = max_val
            elite = i
    if best < .75:
        return 0
    return elite
def quick_crop(im):
    # crop out black bars
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV) # this conversion fails on some images, just ignore bars in this case
    hsv = cv2.blur(hsv,(5,5))
    mask = cv2.inRange(hsv, np.array([0,0,10]), np.array([255,255,255]))
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    if DEBUG:
        d = im.copy()
        cv2.rectangle(d,(x,y),(x+w,y+h),(200,200,0),2)
        # for cnt in contours:
            # x,y,w,h = cv2.boundingRect(cnt)
            # cv2.rectangle(d,(x,y),(x+w,y+h),(200,0,0),2)
        cv2.imshow('quickcrop',mask)
        cv2.waitKey()
    return im[y:y+h,x:x+w]
def parse_squad(path, save_images = True):
    # return list of ops in a card image
    # also generates extra images: cropped image, doctor and risk image
    INTERPOLATION = cv2.INTER_AREA
    INTERPOLATION_ENLARGE = cv2.INTER_CUBIC
    template = cv2.cvtColor(cv2.imread(str(blankTemplate)), cv2.COLOR_BGR2GRAY)
    oim = cv2.imread(str(path))
    if DEBUG:
        cv2.imshow('orig',oim)
        cv2.waitKey()
    print(f'processing {path}...')
    # initial cropping, handles most cases

    oim = quick_crop(oim)
    oim = crop_titlebar(oim)
    oim = crop_sidebars(oim)
    
    if DEBUG:
        cv2.imshow('cropped',oim)
        cv2.waitKey()
        
    h,w = oim.shape[:2]
    # scale to 720
    if h < 720:
        im = cv2.resize(oim, (int(720/oim.shape[0]*oim.shape[1]),720), interpolation = INTERPOLATION_ENLARGE)
    else:
        im = cv2.resize(oim, (int(720/oim.shape[0]*oim.shape[1]),720), interpolation = INTERPOLATION)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # get coordinates of a "Blank" operator.
    wrong_width,right_coords = get_rightmost_blank(im_gray,template)
    if not right_coords:
        #probably failed because image is of "tall" variety, try to crop extra space off top & bottom
        # crop by assume 16:9 and crop center 
        # crop to 16:9 first
        height_mod = int((h-(w/16*9)) / 2)
        if height_mod > 0:
            oim = oim[height_mod:h-height_mod,:]
            h,w = oim.shape[:2]
            if h < 720:
                im = cv2.resize(oim, (int(720/oim.shape[0]*oim.shape[1]),720), interpolation = INTERPOLATION_ENLARGE)
            else:
                im = cv2.resize(oim, (int(720/oim.shape[0]*oim.shape[1]),720), interpolation = INTERPOLATION)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            wrong_width,right_coords = get_rightmost_blank(im_gray,template,force=False)
    rect = None
    for i in range(5):
        if right_coords:
            startX, startY, endX, endY = right_coords
            if wrong_width:
                #setting op width manually.
                endX = startX + template.shape[1]
                rect = (right_coords[0],right_coords[1],template.shape[1],right_coords[3]-right_coords[1])
            else:
                rect = get_blank_bounding_box(right_coords,im,template)
        if not right_coords or abs(rect[2]/rect[3] - OP_ASPECT_RATIO) > .02:
            # if found box is too far off from expected aspect ratio, try again
            wrong_width,right_coords = get_rightmost_blank(im_gray,template,force=True,ignore_rightmost=0)
        else:
            break
            # with open('near_failures.txt','a') as f:
                # f.write(str(path))
                
    # if rect == None this is a failed parse.
    if rect == None:
        with failedParses.open('a') as f:
            f.write(str(path)+'\n')
        return None
    startX, startY, endX, endY = rect[0],rect[1],rect[0]+rect[2],rect[1]+rect[3]
    h,w = rect[3],rect[2]
    # 130x156
    verti_gap = 15+3
    horiz_gap = 9.5
    row_offset = 40
    row_offset_extra = 7 # top row is offset more 
    
    def which_row(im_h, y):
        # return 0-3 for which row this square belongs in.
        # heres coords for each row at 720:
        # 0: 25, 1:196 2:367 3:538
        # 3.4%, 27%, 50.9% 74.7%
        # op list is 691(2?) in height, can divide this by 4 to get coords of rows
        # 173 = 692/4
        op_area_height = 691
        for i in range(4):
            max_y = (op_area_height/4*i) + 25 + (op_area_height/8)
            if y < max_y:
                return i
        return 3
    if DEBUG:
        print('blank template match in row:',which_row(im.shape[0],startY))

    # translate start coords to top right.
    template_row = which_row(im.shape[0],startY)
    startY -= (h+verti_gap) * template_row
    startX += row_offset * template_row
    if template_row != 0:
        startX += row_offset_extra
    startX += w + horiz_gap # move start 1 grid to the right to compensate for bad crop jobs
        
    if DEBUG or SHOW_RES:
        result_disp = im.copy()
    
    def op_at(c,r):
        # converts index to coordinates, (0,0) is top RIGHT and (1,1) is diagonally down left from it.
        new_x = int(startX - w*c - horiz_gap*c - row_offset*r)
        if r != 0:
            new_x -= row_offset_extra
        new_y = startY + h*r + verti_gap*r
        return (new_x, new_y, w, h)
    first_row_found = 0
    rows_remaining = 10
    lowest_row_possible = 3
    c = 0
    operator_list = []
    low_op_clear = False
    crop_border = im.shape[1]
    if DEBUG:
        box = op_at(0,0)
        print('first op at ',box)
        cv2.rectangle(result_disp,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(200,0,200),1)
        cv2.imshow('alignment',result_disp)
        cv2.waitKey()
    
    support_op = None
    # parse every operator from the image:
    while rows_remaining:
        matches = []
        blankcnt = 0
        col_ops = []
        for r in range(3,-1,-1):
            box = op_at(c,r)
            if DEBUG or SHOW_RES:
                cv2.rectangle(result_disp,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(200,0,200),1)
            new_x, new_y = box[:2]
            if box[0] >= im.shape[1]:
                #ENTIRE op is out of bounds, mark this as blank
                matches.append((0,BLANK_NAME,r,0,0))
            else:
                op_img = im[max(box[1],0):box[1]+h,box[0]:box[0]+w]
                expand_by = .5 # 50% but only to the left and bottom, as skill icon is in that corner.
                op_img_expanded = im[int(max(box[1],0)):int(min(box[1]+h*(1+expand_by),im.shape[0])),int(box[0]-w*expand_by):int(box[0]+w)]
                score,name = (match_op(op_img))
                matches.append((score,name,r,match_skill(op_img_expanded, name, im.shape[0]),match_elite(op_img_expanded, name, im.shape[0])))
                if DEBUG or SHOW_RES:
                    cv2.putText(result_disp, name[5:-4], (new_x,new_y+h//3+c*25), cv2.FONT_HERSHEY_SIMPLEX, 
                    .7, (255, 255,0), 2, cv2.LINE_AA)
                    cv2.putText(result_disp, f'{score:.2f}', (new_x,new_y+h//3+c*25+15), cv2.FONT_HERSHEY_SIMPLEX, 
                    .5, (255, 255,0), 2, cv2.LINE_AA)
            if r > lowest_row_possible:
                if matches[-1][1] != BLANK_NAME:
                    # found op below valid area, meaning we are out of bounds.
                    # breaking here saves a lot of processing time
                    break
            if first_row_found and len(matches)>1 and matches[-1][1] == BLANK_NAME and matches[-2][1] != BLANK_NAME:
                #if blank was found above an op (impossible state) then this col is invalid.
                break
        else:
            for score,name,row,skill,elite in matches:
                # if score > .7:
                    # blankcnt = 4
                    # break
                if name == BLANK_NAME:
                    blankcnt+=1
                else:
                    col_ops.append((row,name,skill,elite))
            if (not first_row_found) and len(col_ops)==1:
                try:
                    support_op = [x[1:] for x in col_ops if x[0]>=3][0]
                except IndexError:
                    pass
            rows_remaining -= 1
            if first_row_found:
                lowest_row_possible = min(len(col_ops),3)
            if not first_row_found and blankcnt != 4:
                first_row_found = 1
                lowest_row_possible = min(len(col_ops),3)
                # get rightmost to use for cropping after.
                rmost_row = sorted(col_ops)[0][0]
                rmost_op = op_at(c,rmost_row)
                crop_border = int(rmost_op[0]+rmost_op[2]+horiz_gap)
                if blankcnt < 3:
                    rows_remaining = 2
                elif blankcnt == 3:
                    if col_ops[0][0] == 3:
                        #this is the support op
                        rows_remaining = 3
                        lowest_row_possible = 3
                    else:
                        # this case we need to check for blank==3 each round.
                        rows_remaining = 2
                        low_op_clear = True
            if first_row_found:
                operator_list += [x[1:] for x in col_ops]
            c += 1
            continue
        break
    if DEBUG or SHOW_RES:
        print(operator_list)
    # now crop off right side to get 16:9, BUT not less than crop_border
    # if img is too wide as a result, scale it smaller.
    im = im[:,0:max(1280,crop_border)]
    if DEBUG or SHOW_RES:
        result_disp = result_disp[:,0:max(1280,crop_border)]
    if im.shape[1] > 1280:
        im=cv2.resize(im, (1280,720), interpolation = INTERPOLATION)
        if DEBUG or SHOW_RES:
            result_disp=cv2.resize(result_disp, (1280,720), interpolation = INTERPOLATION)
 
    # convert path to correct img type
    path = path.with_suffix(OUTPUT_IMG_TYPE)
    SQUADS[path.name] = (operator_list,support_op)
    if save_images:
        # cv2.imwrite(str(cropDir.joinpath(path.name)),im)
        
        cv2.imwrite(str(cropDir.joinpath(path.name)),im,[int(cv2.IMWRITE_JPEG_QUALITY), OUTPUT_IMG_QUALITY])
        # cv2.imwrite(str(cropDir.joinpath(path.name)),im,[int(cv2.IMWRITE_WEBP_QUALITY), OUTPUT_IMG_QUALITY])

        risk_coords = (310/720,490/720,90/720,300/720)
        height, width = im.shape[:2]
        risk = im[int(height*risk_coords[0]):int(height*(risk_coords[1])), int(height*risk_coords[2]):int(height*(risk_coords[3]))]
        cv2.imwrite(str(riskDir.joinpath(path.name).with_suffix('.png')),risk)

        nickname_coords = (.37,.41,.2,.5) # y1, y2, x1, x2, multiply all by image height.
        nickname_coords = (.37,.41,.166,.43)
        nickname_coords = (.351,.433,.166,.486)
        nickname_coords = (260/720,305/720,130/720,330/720)
        # nickname_coords = (255/720,310/720,130/720,330/720)
        height, width = im.shape[:2]
        
        nn_box = (int(height*nickname_coords[2]),int(height*nickname_coords[0]),int(height*nickname_coords[3]),int(height*nickname_coords[1]))
        
        if SHOW_RES or DEBUG:
            cv2.rectangle(result_disp,(nn_box[0],nn_box[1]),(nn_box[2],nn_box[3]),(200,200,0),2)
        
        nn = im[int(height*nickname_coords[0]):int(height*(nickname_coords[1])), int(height*nickname_coords[2]):int(height*(nickname_coords[3]))]
        cv2.imwrite(str(doctorDir.joinpath(path.name).with_suffix('.png')),nn)
        
    if DEBUG or SHOW_RES:
        cv2.imshow(path.name,result_disp)
        cv2.waitKey()
        cv2.imwrite('./example_out.png',result_disp)
    return operator_list
    
    
def get_blank_bounding_box(right_coords, im, template):
    startX, startY,endX, endY = right_coords
    blur_hsv = cv2.blur(cv2.cvtColor(im, cv2.COLOR_BGR2HSV),(5,5))
    _,_,im_v = cv2.split(blur_hsv)
    
    seed = (startX+template.shape[1]//4,startY+template.shape[0]//4)
    seedColor = im_v[seed[1],seed[0]]
    
    relevant_only = cv2.inRange(im_v, np.array((seedColor-4,)), np.array((seedColor+4,)))
    mh,mw = relevant_only.shape
    padded = np.zeros((relevant_only.shape[0]+2,relevant_only.shape[1]+2),np.uint8)
    padded[1:mh+1,1:mw+1]=relevant_only

    kernel = np.ones((5,5), np.uint8)
    padded=cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel, iterations=2)

    if DEBUG:
        cv2.imshow('Values',im_v)
        cv2.waitKey()
    _, _, _, rect = cv2.floodFill(im_v,cv2.bitwise_not(padded),seed,255,25,255)
    matchonly = np.zeros((relevant_only.shape[0],relevant_only.shape[1]),np.uint8)
    matchonly[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]] = 255
    rectonly = matchonly.copy() # the full floodfill rect
    _,th = cv2.threshold(im_v,254,255,cv2.THRESH_BINARY)
    matchonly = cv2.bitwise_and(matchonly,th)
    if DEBUG:
        cv2.imshow('matchonly',matchonly)
   
    stragglers = cv2.bitwise_xor(matchonly,rectonly)
    if DEBUG:
        cv2.imshow('stragglers',stragglers)
        cv2.waitKey()
    # remove 1 from the side with the most false pixels
    # find col or row w/ most stragglers, coords are : top right bottom left
    x,y,w,h = rect
    while 1:
        #2nd entry is coord adjustment as x,y,x2,y2
        s_counts = [(np.count_nonzero(stragglers[y:y+1,x:x+w]),(0,1,0,0)),
                    (np.count_nonzero(stragglers[y:y+h,x+w-1:x+w]),(0,0,-1,0)),
                    (np.count_nonzero(stragglers[y+h-1:y+h,x:x+w]),(0,0,0,-1)),
                    (np.count_nonzero(stragglers[y:y+h,x:x+1]),(1,0,0,0)),
                    ]
        errors, adj = max(s_counts)
        if errors > 2:
            x+= adj[0]
            w-= adj[0]
            y+= adj[1]
            h-= adj[1]
            w+= adj[2]
            h+= adj[3]
        else:
            break
    return (x,y,w,h)

#################################
### BUILD AVATAR HISTOGRAMS #####
#################################
AVATAR_CROP = .9   
BLANK_NAME = "char_BLANK.png"
G_SIFT = cv2.SIFT_create()
G_BF = cv2.BFMatcher()
AV_HISTS_DATA = Path('./av_hists.p').resolve()
AV_SIFTS_DATA = Path('./av_sifts.p').resolve()
OPMASK_PATH = Path('./ALL_OPS_MASK.png').resolve()
def _generate_avatar_data(rebuild_all=False):
    global av_hists, av_sifts, OPMASK, BLANK_HIST
    if not rebuild_all and (AV_HISTS_DATA.exists() and AV_SIFTS_DATA.exists() and OPMASK_PATH.exists()):
        with AV_HISTS_DATA.open('rb') as f:
            av_hists = pickle.load(f)
        with AV_SIFTS_DATA.open('rb') as f:
            av_sifts = pickle.load(f)
        OPMASK = cv2.imread(str(OPMASK_PATH), cv2.IMREAD_UNCHANGED)
    else:
        # create grayscale background:
        ramp_width,g_h = 180,180
        rampr = np.linspace(78, 48, ramp_width).astype(np.uint8)
        rampr = np.tile(np.transpose(rampr), (g_h,1))
        rampr = cv2.merge([rampr,rampr,rampr])

        av_hists = []
        avatars = list(avatarDir.glob('char_*'))#[:1]
        
        # remove invalid files like "char_298_susuro5.png" or "char_340_shwazr6.png"
        invalid_ext = re.compile('\d+(\..+)')
        all_names = set([a.name for a in avatars])
        avatars = [av for av in avatars if invalid_ext.sub('\\1',av.name) not in all_names.difference([av.name])]
        avatars = [av for av in avatars if av.name not in ('char_1001_amiya2_2.png',)]
        # create OPMASK, a bitwise and of ALL operator images.
        ret, OPMASK = cv2.threshold(cv2.imread(str(next(iter(avatars))), cv2.IMREAD_UNCHANGED)[:, :, 3], 0, 255, cv2.THRESH_BINARY)
        for av in avatars:
            im = cv2.imread(str(av), cv2.IMREAD_UNCHANGED)
            ret, mask = cv2.threshold(im[:, :, 3], 0, 255, cv2.THRESH_BINARY)
            OPMASK = cv2.bitwise_and(mask, OPMASK)
        OPMASK = smoothimg(OPMASK)
        kernel = np.ones((5,5), np.uint8)
        OPMASK = cv2.dilate(OPMASK, kernel, iterations=2)
        cv2.imwrite(str(OPMASK_PATH), OPMASK)

        
        av_sifts = {}
        for av in avatars:
            l_img = rampr.copy()
            s_img = cv2.imread(str(av),cv2.IMREAD_UNCHANGED)#was -1 
            
            av_sifts[av.name] = G_SIFT.detectAndCompute(cv2.cvtColor(s_img, cv2.COLOR_BGR2GRAY), None)
            ret, alphamask = cv2.threshold(s_img[:, :, 3], 0, 255, cv2.THRESH_BINARY)
            x_offset,y_offset = 0,0
            y1, y2 = y_offset, y_offset + s_img.shape[0]
            x1, x2 = x_offset, x_offset + s_img.shape[1]

            alpha_s = s_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                          alpha_l * l_img[y1:y2, x1:x2, c])
            h,w,_ = l_img.shape
            im = l_img
            hist_base = avatar_hist(im,True)
            av_hists.append((hist_base,av,im))

        # add blank image to both
        blank = cv2.imread(str(blankTemplate), cv2.IMREAD_UNCHANGED)
        blank = blank[0:blank.shape[1],:]
        hist_base = avatar_hist(blank,True,False)
        av_hists.append((hist_base,avatarDir.joinpath(BLANK_NAME),blank))
        av_sifts[BLANK_NAME] = G_SIFT.detectAndCompute(cv2.cvtColor(cv2.imread(str(blankTemplate), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY), None)
    BLANK_HIST = av_hists[-1][0]
    with AV_HISTS_DATA.open('wb') as f:
        pickle.dump(av_hists, f)
    with AV_SIFTS_DATA.open('wb') as f:
        pickle.dump(av_sifts, f)
        
DEBUG = False
SHOW_RES = False
DO_ASSERTS = False

if __name__ == '__main__':
    update_char_table()
    if TAG not in CC_START_DATES:
        print('invalid cc tag')
        exit(1)
    for dpath in (doctorDir,cropDir,riskDir,numsDir,imagesDir,thumbsDir):
        dpath.mkdir(exist_ok=True)
        # create duplicates folders
    for dir in (cropDir,doctorDir,thumbsDir):
        dir.joinpath('duplicates/').mkdir(exist_ok=True)
    CCSTART = CC_START_DATES.get(TAG,0)
    # cc_week_2 = (CCSTART + 604800) - (CCSTART % (60 * 60 * 24)) + 39600
    # cc_day_1 = (CCSTART + 172800) - (CCSTART % (60 * 60 * 24)) + 39600
    with charDataPath.open('rb') as f:
        CHAR_DATA = json.load(f)
    SKILL_ICONS = {}
    for sk in skillDir.glob('skill_*.png'):
        SKILL_ICONS[sk.name[11:-4]] = cv2.imread(str(sk))
    ELITE_ICONS = {}
    for sk in eliteDir.glob('?_s_box.png'):
        # icons have varying white space
        if 1 or sk.name[0] == '1':
            ELITE_ICONS[sk.name[0]] = cv2.imread(str(sk),cv2.IMREAD_UNCHANGED)#[(256-176):256,0:262]
            b_channel, g_channel, r_channel, a_channel = cv2.split(ELITE_ICONS[sk.name[0]])
            blank_image = np.zeros((*ELITE_ICONS[sk.name[0]].shape[:2],3), np.uint8)
            # print(blank_image.shape, ELITE_ICONS[sk.name[0]][:,:,:].shape)
            # im = cv2.bitwise_and(blank_image, ELITE_ICONS[sk.name[0]])
            ELITE_ICONS[sk.name[0]] = cv2.bitwise_and(ELITE_ICONS[sk.name[0]], ELITE_ICONS[sk.name[0]],mask = a_channel)[:,:,:3]
            # print('INIT SHAPE',ELITE_ICONS[sk.name[0]].shape)
            # cv2.imshow('elite',ELITE_ICONS[sk.name[0]][:,:,:3])
            # cv2.waitKey(0)
            # exit()
        # if sk.name[0] == '2':
            # ELITE_ICONS[sk.name[0]] = cv2.imread(str(sk),cv2.IMREAD_UNCHANGED)#[3:37,0:40]#[(256-176):256,0:262]
    
    # fix for broken skill icons:
    with skillDataPath.open('r') as f:
        for k,v in json.load(f).items():
            # print('icon for',k,v.get('iconId',v['skillId']))
            if not k.startswith('sktok_'):
                SKILL_ICONS[k] = SKILL_ICONS.get(k, SKILL_ICONS[v['iconId'] or v['skillId']])

    # for char in CHAR_DATA:
        # for sk in CHAR_DATA[char]['skills']:
            # if sk['skillId'] not in SKILL_ICONS:
                # print(sk['skillId'])
    # exit()
    test = None
    paths = list(imagesDir.glob('*.*'))
    # test a random image:
    # paths = [random.choice(paths)]
    # test a specific image:
    # test = './images-cc6clear/1646131485278.jpg' 
   
    # test = './images-cc1clear/1605109000511.jpg' 
    # test = './images-cc1clear/1606203199640.jpg'
    # test = './images-cc4clear/1627301022848.png'
    # test = './images-cc3clear/1622356297879.png'
    # test = './images-cc4clear/1626269984534.png'
    # test = './images-cc4clear/1626719849150.jpg'
    # test = './images-cc2clear/1613049415748.png'
    # test = './images-cc1clear/1605109000511.jpg'
    # test = './images-cc5clear/1636642844861.jpg' # sunglasses failed
    # test = './images-cc5clear/1636882207237.png' # joker edit
    
    # test = './images-cc4clear/1626720793105.png' #meteorite recognized as bena.
    # test = './images-cc0clear/1599836698409.jpg' #meteorite recognized as bena.
    # test = './images-cc3clear/1622437965161.png' # multiple skills wrong
    # test = './images-cc6clear/1646344197575.jpg' # img missing
    # test = './images-cc6clear/1647236118113.png' # img missing
    # test = r'.\images-cc7clear\1655318875062.jpg'
    # test = './images-cc7clear/1655362131273.png'
    # test = './images-cc7clear/1655503623630.png'
    # test = './images-cc1clear/1605850911232.png'
    # test = './images-cc8clear/1662131179112.jpg' # nian (support) recog as plat -- UNFIXED -- ( see match_op(), the line best[0] > .108, this image requires .102, but this breaks oather images.
    # test = './images-cc2clear/1613547301651.png' # blurred op images causes failure
    # test = './images-cc3clear/1622485108107.jpg'
    # test = './images-cc3clear/1622485108107.jpg'
    # DEBUG = True
    # SHOW_RES = True
    # DO_ASSERTS = True

    if test:
        paths = [Path(test).resolve()]
    # if len(paths) == 1:
        SHOW_RES = True

    with assertTests.open('rb') as f:
        assert_pairs = pickle.load(f)
    # assert_pairs = 
    from pprint import pprint
    # add new assert tests (this is an example):
    assert_pairs['./images-cc4clear/1626450296491.jpg']: ['char_002_amiya_winter#1.png',
                                         'char_141_nights_2.png',
                                         'char_150_snakek_2.png',
                                         'char_271_spikes_winter#2.png',
                                         'char_286_cast3.png',
                                         'char_237_gravel_winter#2.png',
                                         'char_258_podego_2.png',
                                         'char_151_myrtle_2.png',
                                         'char_278_orchid.png',
                                         'char_196_sunbr_summer#1.png',
                                         'char_272_strong.png',
                                         'char_345_folnic_2.png']
    with open('asserts_tests_pairs.txt','w') as f:
        pprint(assert_pairs,f)
    with assertTests.open('wb') as f:
        pickle.dump(assert_pairs, f)
    # must do this before anything else
    # to ensure data isn't stale, check file mtime (stale is over 14 days old)
    _generate_avatar_data(rebuild_all = (AV_HISTS_DATA.exists() and (time.time() - AV_HISTS_DATA.stat().st_mtime > 60*60*24*14)))

    if DO_ASSERTS and not SHOW_RES:
        fails = 0
        for f,o in assert_pairs.items():
            squad = set([x[0] for x in parse_squad(Path(f).resolve(),save_images=False)])
            try:
                assert squad == set(o)
            except:
                print('failed for img',f,'expected:')
                pprint(set(o))
                print('got:')
                pprint(squad)
                fails+=1
                raise
        if fails:
            print(f'{fails} ASSERTS FAILED '*20)
        else:
            print('ALL ASSERTS PASSED '*20)
        exit()
    if 0:
        # test dupe finder
        # document.querySelectorAll('.cardContainer').forEach(c=> {
            # if (!c.getAttribute('data-dupe'))
                # c.style.display='none'
        # })
        if TAG != '-ccbclear':
            with DATA_JSON.open('r') as f:
                data = json.load(f)
            remove_duplicates(data)
            # with DATA_JSON.open('w') as f:
                # json.dump(data,f)
        exit()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_url = {executor.submit(parse_squad, path, test is None): path for path in paths}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
            except:
                print('EXCEPTION IN THREAD:',url,flush=True)
                with failedParses.open('a') as f:
                    f.write(str(url)+'\n')
                raise
    # if len(paths)==1:
    if test:
        exit()
    with SQUAD_JSON.open('w') as f:
        json.dump(SQUADS,f)
    print('processing done...')
    generate_data_json()
    with DATA_JSON.open('r') as f:
        data = json.load(f)
    print('creating thumbnails...')
    createThumbs()
    print('parsing risks...')
    parse_risks(data)
    fix_json_data(data)
    print('removing dupes...')
    if TAG != '-ccbclear':
        remove_duplicates(data)
        group_duplicates(data)
    add_extra_data(data)
    with DATA_JSON.open('w') as f:
        json.dump(data,f)
    # calculate_soul(data) # this is now done in the js, client side
    
    print('cleaning output dirs')
    clean_output_folders(data)
    with DATA_JSON.open('w') as f:
        json.dump(data,f)
        
    # merge all data files into one combined -cc-all
    full = {}
    for path in JSON_DIR.glob('data-*clear.json'):
        with path.open('r') as f:
            dictupdate(full, {k:dict(v,**{'tag':path.name[4:].split('.')[0]}) for k,v in json.load(f).items()})
    with open(JSON_DIR.joinpath('./data-cc-all.json'), 'w') as f:
        json.dump(full, f)


# risk parse test
# import copy
# with DATA_JSON.open('r') as f:
    # data = json.load(f)
# print('parsing risks...')
# parsed = parse_risks(copy.deepcopy(data))
# print(len(parsed),len(data))
# fix_json_data(data)
# for k in data:
    # if parsed[k]['risk'] != data[k]['risk']:
        # print(k,parsed[k]['risk'], data[k]['risk'])
