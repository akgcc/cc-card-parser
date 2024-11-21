'''Get banner history from gamepress and convert to json format.'''
## NOTE: if gamepress revives check to make sure kernal locating banners are ignored
import requests
import re
import json
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
get_operator_lists(live=True)
NA_OPS = {}
get_operator_lists_wiki()
with open('./json/banner_history.json','w') as f:
    if NA_OPS and CN_OPS:
        json.dump({'NA':NA_OPS,'CN':CN_OPS},f)
