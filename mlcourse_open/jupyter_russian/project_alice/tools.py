import pandas as pd
import numpy as np
import re

def write_to_submission_file(predicted_labels, out_file, target='target', index_label='session_id'):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])

    predicted_df.to_csv(out_file, index_label=index_label)


def get_domen(url):
    if re.search('[a-zA-Z]', url):
        return url.split('.')[-1]
    return 'ip'


def get_part_of_day(x):
    x = int(x.hour)
    if (0 <= x < 12):
        return 0
    elif (12 <= x < 15):
        return 1
    elif (15 <= x < 18):
       return 2
    elif (18 <= x < 21):
       return 3
    else:
       return 4


def one_site_mean_duration(x):
    site_times = [datetime for datetime in list(x) if not pd.isnull(datetime)]
    durations = [site_times[i] - site_times[i-1] for i in range(1, len(site_times))]
    durations = list(map(lambda x: x.seconds, durations))

    if (len(durations) > 0):
        return np.mean(durations)

    return 0


def count_alice_top_sites(top_sites, x):
    x_list = list(x)
    x_list = [site for site in x_list if site in top_sites]

    return len(x_list)