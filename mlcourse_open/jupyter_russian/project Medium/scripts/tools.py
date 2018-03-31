import json
import os
import pandas as pd
from html.parser import HTMLParser
from textblob import TextBlob


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def read_json_line(line=None):
    result = None
    try:
        result = json.loads(line)
    except Exception as e:
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = ' '
        new_line = ''.join(new_line)
        return read_json_line(line=new_line)
    return result


def write_submission_file(prediction, filename, path_to_sample=os.path.join('../medium data/', 'sample_submission.csv')):
    submission = pd.read_csv(path_to_sample, index_col='id')

    submission['log_recommends'] = prediction
    submission.to_csv(filename)


def translate_dataframe_to_english(dataframe):
    list_of_translated = list()
    logs = list()

    for i in range(dataframe.shape[0]):
        blob = TextBlob(dataframe.iloc[i, 0])

        if (blob.detect_language() != 'en'):
            try:
                list_of_translated.append(str(blob.translate(to="en")))
                print(i)
            except:
                list_of_translated.append(dataframe.iloc[i, 0])
                print('{0} - translate error'.format(i))
                logs.append(i)

        else:
            list_of_translated.append(dataframe.iloc[i, 0])
            print('{0} - already in english'.format(i))

    dataframe_translated = pd.DataFrame(list_of_translated)
    return dataframe_translated


def translate_to_english(text):

    blob = TextBlob(text)
    out = ''
    if (blob.detect_language() != 'en'):
        try:
            out = str(blob.translate(to="en"))
        except:
            out = text
    else:
        out = text
    return out






























