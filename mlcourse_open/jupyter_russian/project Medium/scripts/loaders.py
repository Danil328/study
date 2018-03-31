# -*- coding: utf-8 -*-

import pandas as pd
import tqdm
from tqdm import tqdm
from tools import read_json_line
from tools import strip_tags

# ids        да
# spider     полность нет, 1 значение
# timestamp  наверное
# author     да
# content    да
# domain     да
# image_url  наверно
# link_tags  наверно
# meta_tages наверно
# published  да
# tags       полностью нет, 1 пустое значение
# title      да
# url        наверно



def load_ids(path_to_inp_json_file):
    output_list = list()
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
            json_data = read_json_line(line)
            id = json_data['_id']
            output_list.append(id)

    return pd.DataFrame({'id': output_list})

def load_timestamps(path_to_inp_json_file):
    output_list = list()
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
            json_data = read_json_line(line)
            timestamp = json_data['_timestamp']
            output_list.append(timestamp)

    return pd.DataFrame({'timestamp': output_list})

def load_authors(path_to_inp_json_file):
    output_list = list()
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
            json_data = read_json_line(line)
            author = json_data['author']
            output_list.append(author)

    # name
    return pd.DataFrame.from_dict(output_list)[['twitter', 'url']]

def load_contents(path_to_inp_json_file):
    output_list = []
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
            json_data = read_json_line(line)
            content = json_data['content'].replace('\n', ' ').replace('\r', ' ')
            content_no_html_tags = strip_tags(content)
            output_list.append(content_no_html_tags)
    return pd.DataFrame(output_list)

def load_domains(path_to_inp_json_file):
    output_list = list()
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
            json_data = read_json_line(line)
            domain = json_data['domain']
            output_list.append(domain)

    return pd.DataFrame({'domain': output_list})

def load_image_urls(path_to_inp_json_file):
    output_list = list()
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
            json_data = read_json_line(line)
            image_url = json_data['image_url']
            output_list.append(image_url)

    return pd.DataFrame({'image_url': output_list})

def load_link_tags(path_to_inp_json_file):
    output_list = list()
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
            json_data = read_json_line(line)
            link_tags = json_data['link_tags']
            output_list.append(link_tags)

    #drop icon, mask-icon, publisher, search
    return pd.DataFrame.from_dict(output_list)[['alternate', 'amphtml', 'apple-touch-icon', 'author', 'canonical', 'stylesheet']]

def load_meta_tags(path_to_inp_json_file):
    output_list = list()
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
            json_data = read_json_line(line)
            meta_tags = json_data['meta_tags']
            output_list.append(meta_tags)

    useless_columns = ['al:android:app_name', 'al:android:package', 'al:ios:app_name', 'al:ios:app_store_id',
                       'fb:app_id', 'og:type', 'theme-color', 'twitter:app:id:iphone', 'twitter:app:name:iphone',
                       'twitter:label1', 'viewport']
    output_df = pd.DataFrame.from_dict(output_list)
    return output_df[output_df.columns.difference(useless_columns)]

def load_publisheds(path_to_inp_json_file):
    output_list = list()
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
            json_data = read_json_line(line)
            published = json_data['published']
            output_list.append(published['$date'])

    return pd.DataFrame({'published_$date': output_list})

def load_titles(path_to_inp_json_file):
    output_list = list()
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
            json_data = read_json_line(line)
            title = json_data['title']
            output_list.append(title)

    return pd.DataFrame({'title': output_list})

def load_urls(path_to_inp_json_file):
    output_list = list()
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
            json_data = read_json_line(line)
            url = json_data['url']
            output_list.append(url)

    return pd.DataFrame({'url': output_list})
