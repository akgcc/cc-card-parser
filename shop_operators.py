'''Get banner history from gamepress and convert to json format.'''
## NOTE: if gamepress revives check to make sure kernal locating banners are ignored
import requests
import re
import json
from pprint import pprint
from urllib.parse import quote
from html import unescape
# import xml.etree.ElementTree as ET
# from lxml import etree
NA_OPS = {}
CN_OPS = {}
def get_operator_lists(live = True):
    # xml is malformed, so we use regex instead.
    if live:
        r = requests.get('https://gamepress.gg/arknights/database/banner-list-gacha')
        r.encoding = 'utf8'
        content = r.text
        with open('i_oplist_cache','w',encoding='utf8') as f:
            f.write(content)
    else:
        with open('i_oplist_cache','r',encoding='utf8') as f:
            content = f.read()
    tables = re.compile('<tbody>.*?</tbody>',re.DOTALL)
    # skip the first table as it's just upcoming for global
    for table in tables.findall(content)[1:]:
        rows = re.compile('<tr>.*?</tr>',re.DOTALL)
        for row in rows.findall(table):
            cn_date = re.compile('<td[^>]*views-field-field-cn-end-date[^>]*>(.*?)</td>',re.DOTALL)
            na_date = re.compile('<td[^>]*views-field-field-start-time[^>]*>(.*?)</td>',re.DOTALL)
            shop_ops = re.compile('<td[^>]*views-field-field-store-operators[^>]*>(.*?)</td>',re.DOTALL)
            banner_ops = re.compile('<td[^>]*views-field-field-featured-characters[^>]*>(.*?)</td>',re.DOTALL)
            time_parser = re.compile('<time[^>]*?>([^<]*)</time>',re.DOTALL)
            op_parser = re.compile('<a[^>]*?>([^<]*)</a>',re.DOTALL)
            link_parser = re.compile('<td[^>]*views-field-field-event-banner[^>]*>[^<]*<a\\s+href="([^"]*)',re.DOTALL)
            na_times = na_date.findall(row)[0]
            cn_times = cn_date.findall(row)[0]
            na_m = time_parser.match(na_times)
            cn_m = time_parser.match(cn_times)
            sim_link = link_parser.findall(row)[0]
            blue = int('#BB_'.lower() in sim_link.lower())
            for op_name in op_parser.findall(shop_ops.findall(row)[0]):
                if na_m:
                    l = NA_OPS.setdefault(op_name,{'shop':[],'banner':[]})
                    l['shop'].append({'date': na_m.group(1), 'blue': blue})
                if cn_m:
                    l = CN_OPS.setdefault(op_name,{'shop':[],'banner':[]})
                    l['shop'].append({'date': cn_m.group(1), 'blue': blue})
            for op_name in op_parser.findall(banner_ops.findall(row)[0]):
                if na_m:
                    l = NA_OPS.setdefault(op_name,{'shop':[],'banner':[]})
                    l['banner'].append({'date': na_m.group(1), 'blue': blue})
                if cn_m:
                    l = CN_OPS.setdefault(op_name,{'shop':[],'banner':[]})
                    l['banner'].append({'date': cn_m.group(1), 'blue': blue})
def extract_banners_cell_templates(wikitext):
    # chatGPT function
    # Regex pattern to match {{Banners cell|...}} templates
    pattern = r'{{Banners cell\|([^}]+)}}'

    # Find all matches
    matches = re.findall(pattern, wikitext)

    # List to store parsed parameters for each template
    parsed_templates = []

    for match in matches:
        # Split parameters by '|' and extract key-value pairs
        parameters = {}
        for param in match.split('|'):
            # Split key-value by '='
            key_value = param.split('=', 1)
            if len(key_value) == 2:
                parameters[key_value[0].strip()] = key_value[1].strip()
        parsed_templates.append(parameters)

    return parsed_templates
def extract_prts_banners_cell_templates(wikitext):
    # Another chatGPT function
    # Regex to match each row
    row_pattern = re.compile(
    r'\|-\s*\n'                    # Match row start '|-' followed by optional spaces and a newline
    r'(?:\|.*?\s*)*?'              # Skip any cells before the date, non-capturing group
    r'\|(\d{4}-\d{2}-\d{2}).*?'  # Match and capture the date (format: YYYY-MM-DD)
    r'(\|.*?)(?=\|-\s*\n|\Z)',       # Capture everything after the date until the next row starts or end of content
    re.DOTALL
)
    # Regex to match the {{干员头像}} templates within a cell
    avatar_pattern = re.compile(r'\{\{干员头像\|([^|}]+)(.*?)\}\}')

    results = []
    for match in row_pattern.finditer(wikitext):
        date = match.group(1).strip()
        avatar_cells = match.group(2).strip()
        # Extract and parse all {{干员头像}} templates
        avatars = []
        for avatar_match in avatar_pattern.finditer(avatar_cells):
            name = avatar_match.group(1).strip()  # The name of the avatar
            args = avatar_match.group(2).strip()  # Any optional arguments
            parsed_args = {}
            if args:
                # Split optional arguments and parse them as key-value pairs
                for arg in args.split('|'):
                    if arg:
                        if '=' in arg:
                            key, value = arg.split('=', 1)
                            parsed_args[key.strip()] = value.strip()
                        else:
                            parsed_args[arg.strip()] = None
            avatars.append({"name": name, "args": parsed_args})
        results.append({
            "date": date,
            "avatars": avatars,
        })
    return results
def get_operator_lists_wiki():
    # get list of banner pages:
    r = requests.get('https://arknights.wiki.gg/wiki/Category:Former_headhunting_banners')
    r.encoding = 'utf8'
    content = r.text
    pages = re.compile('href="/wiki/Headhunting/Banners/Former([^"]*)',re.DOTALL)
    urls = ['https://arknights.wiki.gg/wiki/Headhunting/Banners?action=edit'] # current banners
    urls.extend([f'https://arknights.wiki.gg/wiki/Headhunting/Banners/Former{suffix}?action=edit' for suffix in pages.findall(content)])
    for url in urls:
        r = requests.get(url)
        r.encoding = 'utf8'
        content = r.text
        match = re.search(r'<textarea[^>]*>(.*?)</textarea>', content, re.DOTALL)
        mediawiki_source = match.group(1)
        banners = []
        banners.extend(extract_banners_cell_templates(mediawiki_source)) # need to do this for each year
        
        for banner in banners:
            blue = int('type' in banner and 'kernal' not in banner['type'].lower())
            if blue and 'locating' in banner['type'].lower():
                continue # skip kernal locating
            date = banner['date'].split('&amp;ndash;')[0].strip() if 'date' in banner else banner['global'].split('&amp;ndash;')[0].strip()
            ops = []
            for key in ['operators','operators1','operators2']:
                if key in banner:
                    ops.extend([n.strip() for n in banner[key].split(',')])
            store = [int(n.strip()) for n in banner['store'].split(',')] if 'store' in banner else [0]*len(ops)
            for i, op_name in enumerate(ops):
                l = NA_OPS.setdefault(op_name,{'shop':[],'banner':[]})
                l['banner'].append({'date': date, 'blue': blue})
                if store[i] == 1:
                    l['shop'].append({'date': date, 'blue': blue})
def get_operator_lists_prts():
    # list of blue banner years:
    r = requests.get('https://prts.wiki/index.php?title=%E5%8D%A1%E6%B1%A0%E4%B8%80%E8%A7%88/%E5%B8%B8%E9%A9%BB%E4%B8%AD%E5%9D%9A%E5%AF%BB%E8%AE%BF%26%E4%B8%AD%E5%9D%9A%E7%94%84%E9%80%89&action=edit')
    r.encoding = 'utf8'
    content = r.text
    match = re.search(r'<textarea[^>]*>(.*?)</textarea>', content, re.DOTALL)
    mediawiki_source = match.group(1)
    pages = re.findall('pageName=([^|}]*)',mediawiki_source)
    blue = len(pages)
    # list of standard banner years:
    r = requests.get('https://prts.wiki/index.php?title=%E5%8D%A1%E6%B1%A0%E4%B8%80%E8%A7%88/%E5%B8%B8%E9%A9%BB%E6%A0%87%E5%87%86%E5%AF%BB%E8%AE%BF&action=edit')
    r.encoding = 'utf8'
    content = r.text
    match = re.search(r'<textarea[^>]*>(.*?)</textarea>', content, re.DOTALL)
    mediawiki_source = match.group(1)
    pages.extend(re.findall('pageName=([^|}]*)',mediawiki_source))
    # these are limited banners, which include discontinued ones (like solo rate ups), currently they are just on one page
    pages.append('卡池一览/限时寻访')
    for page in pages:
        url = f'https://prts.wiki/index.php?title={quote(unescape(page))}&action=edit'
        r = requests.get(url)
        r.encoding = 'utf8'
        content = r.text
        match = re.search(r'<textarea[^>]*>(.*?)</textarea>', content, re.DOTALL)
        mediawiki_source = match.group(1)
        banners = []
        banners.extend(extract_prts_banners_cell_templates(mediawiki_source)) # need to do this for each year
        for banner in banners:
            # skip kernal locating ??
            date = banner['date'].split('~&lt;')[0].strip()
            for op in banner['avatars']:
                l = CN_OPS.setdefault(op['name'],{'shop':[],'banner':[]})
                l['banner'].append({'date': date, 'blue': int(blue > 0)})
                if 'shop' in op['args'] or 'shop2' in op['args']:
                    l['shop'].append({'date': date, 'blue': int(blue > 0)})
            blue -= 1
get_operator_lists_prts()
# get_operator_lists(live=True)
get_operator_lists_wiki()
with open('./json/banner_history.json','w') as f:
    if NA_OPS and CN_OPS:
        json.dump({'NA':NA_OPS,'CN':CN_OPS},f)
