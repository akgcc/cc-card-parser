'''Get banner history from gamepress and convert to json format.'''
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
            link_parser = re.compile('<td[^>]*views-field-field-event-banner[^>]*>[^<]*<a\s+href="([^"]*)',re.DOTALL)
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
                    
get_operator_lists(live=True)
with open('./json/banner_history.json','w') as f:
    if NA_OPS and CN_OPS:
        json.dump({'NA':NA_OPS,'CN':CN_OPS},f)
