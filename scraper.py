import sys
from requests import Session
from lxml import html as lxmlhtml
from pathlib import Path
import time

tag = sys.argv[1]
base_url = f'https://arch.b4k.co/vg/search/filename/%2A{tag}%2A/'

outputDir = Path(f'./images{tag}/').resolve()
outputDir.mkdir(exist_ok=True)
s=Session()

next_page = base_url
total = 0
while next_page:
    with s.get(next_page) as r:
        etree = lxmlhtml.fromstring(r.text)
    images = etree.xpath('//aside[contains(@class,"posts")]/article//div[contains(@class,"post_file")]/a/@href')
    next_page = next(iter(etree.xpath('//li[@class="next"]/a/@href')),None)
    new = 0
    for imagepath in images:
        destPath = outputDir.joinpath(imagepath.split('/')[-1])
        dupePath = outputDir.joinpath('invalid/').joinpath(imagepath.split('/')[-1])
        print(destPath)
        if not destPath.exists() and not dupePath.exists():
            new += 1
            with destPath.open('wb') as f:
                f.write(s.get(imagepath).content)
                total += 1
            time.sleep(1)
    else:
        if not new and len(images) > 5:
            break
        time.sleep(2)
        continue
    break
if total == 0:
    exit(0)