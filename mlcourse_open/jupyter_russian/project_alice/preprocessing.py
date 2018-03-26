
import pickle
import pandas as pd
import numpy as np
from tools import get_domen, get_part_of_day, one_site_mean_duration, count_alice_top_sites
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def base_preprocessing(train, test):

    times = ['time%s' % i for i in range(1, 11)]
    train[times] = train[times].apply(pd.to_datetime)
    test[times] = test[times].apply(pd.to_datetime)

    train = train.sort_values(by='time1')

    sites = ['site%s' % i for i in range(1, 11)]
    train[sites] = train[sites].fillna(0).astype('int')
    test[sites] = test[sites].fillna(0).astype('int')

    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    return X_train, test, y_train

def benchmark_1_preprocessing(X_train, X_test):
    '''
    CountVectorizer
    '''

    sites = ['site%s' % i for i in range(1, 11)]
    X_train_sites_only = X_train[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)
    X_test_sites_only = X_test[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)

    count_vectorizer = CountVectorizer()
    count_vectorizer.fit(X_train_sites_only)

    X_train_bow = count_vectorizer.transform(X_train_sites_only)
    X_test_bow = count_vectorizer.transform(X_test_sites_only)

    return X_train_bow, X_test_bow

def benchmark_2_preprocessing(X_train, X_test):
    '''
    CountVectorizer + start seession time (yyyymm)
    '''

    # TfidfVectorizer
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_sites_only = X_train[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)
    X_test_sites_only = X_test[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)

    count_vectorizer = CountVectorizer().fit(X_train_sites_only)

    train_bow = count_vectorizer.transform(X_train_sites_only)
    test_bow = count_vectorizer.transform(X_test_sites_only)


    # Start seession time (yyyymm)
    start_session_time_train = pd.DataFrame(X_train['time1'].apply(lambda x: 100 * x.year + x.month))
    start_session_time_test = pd.DataFrame(X_test['time1'].apply(lambda x: 100 * x.year + x.month))

    scaler = StandardScaler().fit(start_session_time_train)

    start_session_time_train = scaler.transform(start_session_time_train)
    start_session_time_test = scaler.transform(start_session_time_test)


    # Final concat
    X_train = csr_matrix(hstack([train_bow, start_session_time_train]))
    X_test = csr_matrix(hstack([test_bow, start_session_time_test]))

    return X_train, X_test

def benchmark_3_preprocessing(X_train, X_test):
    '''
    TfidfVectorizer + start session time (yyyymm) + start_hour + morning
    '''
    #TfidfVectorizer
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_sites_only = X_train[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)
    X_test_sites_only = X_test[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X_train_sites_only)

    train_tfidf = tfidf_vectorizer.transform(X_train_sites_only)
    test_tfidf = tfidf_vectorizer.transform(X_test_sites_only)


    # Start session time (yyyymm)
    start_session_time_train = pd.DataFrame(X_train['time1'].apply(lambda x: 100 * x.year + x.month))
    start_session_time_test = pd.DataFrame(X_test['time1'].apply(lambda x: 100 * x.year + x.month))

    scaler = StandardScaler().fit(start_session_time_train)

    start_session_time_train = scaler.transform(start_session_time_train)
    start_session_time_test = scaler.transform(start_session_time_test)


    # Start hour
    start_hour_train = pd.DataFrame(X_train['time1'].apply(lambda x: x.hour))
    start_hour_test = pd.DataFrame(X_test['time1'].apply(lambda x: x.hour))

    scaler = StandardScaler().fit(start_hour_train)

    start_hour_train = scaler.transform(start_hour_train)
    start_hour_test = scaler.transform(start_hour_test)


    # Start morning
    binary_morning_train = pd.DataFrame(X_train['time1'].apply(lambda x: 1 if x.hour <= 11 else 0))
    binary_morning_test = pd.DataFrame(X_test['time1'].apply(lambda x: 1 if x.hour <= 11 else 0))


    # Final concat
    X_train = csr_matrix(hstack([train_tfidf, start_session_time_train, start_hour_train, binary_morning_train]))
    X_test = csr_matrix(hstack([test_tfidf, start_session_time_test, start_hour_test, binary_morning_test]))

    return X_train, X_test

def variant_1_preprocessing(X_train, X_test):
    '''
    TfidfVectorizer
    '''

    sites = ['site%s' % i for i in range(1, 11)]
    X_train_sites_only = X_train[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)
    X_test_sites_only = X_test[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X_train_sites_only)

    train_tfidf = tfidf_vectorizer.transform(X_train_sites_only)
    test_tfidf = tfidf_vectorizer.transform(X_test_sites_only)


    return train_tfidf, test_tfidf

def variant_2_preprocessing(X_train, X_test):
    '''
    TfidfVectorizer + domens CountVectorizer
    '''

    #TfidfVectorizer
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_sites_only = X_train[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)
    X_test_sites_only = X_test[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X_train_sites_only)

    train_tfidf = tfidf_vectorizer.transform(X_train_sites_only)
    test_tfidf = tfidf_vectorizer.transform(X_test_sites_only)


    # Domens
    with open(r'data/site_dic.pkl', 'rb') as input_file:
        site_dict = pickle.load(input_file)

    invert_site_dict = {v: k for k, v in site_dict.items()}
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_domens_only = X_train[sites].applymap(lambda x: get_domen(invert_site_dict[x]) if x in invert_site_dict else 'nan')
    X_test_domens_only = X_test[sites].applymap(lambda x: get_domen(invert_site_dict[x]) if x in invert_site_dict else 'nan')

    X_train_domens_only = X_train_domens_only[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)
    X_test_domens_only = X_test_domens_only[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)

    domens_tfidf_vectorizer = CountVectorizer()
    domens_tfidf_vectorizer.fit(X_train_domens_only)

    domens_train_tfidf = domens_tfidf_vectorizer.transform(X_train_domens_only)
    domens_test_tfidf = domens_tfidf_vectorizer.transform(X_test_domens_only)


    # Final concat
    X_train = csr_matrix(hstack([train_tfidf, domens_train_tfidf]))
    X_test = csr_matrix(hstack([test_tfidf, domens_test_tfidf]))


    return X_train, X_test

def variant_3_preprocessing(X_train, X_test):
    '''
    TfidfVectorizer + start session time (yyyymm) + start_hour + morning + domens CountVectorizer
    '''
    #TfidfVectorizer
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_sites_only = X_train[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)
    X_test_sites_only = X_test[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X_train_sites_only)

    train_tfidf = tfidf_vectorizer.transform(X_train_sites_only)
    test_tfidf = tfidf_vectorizer.transform(X_test_sites_only)


    # Start session time (yyyymm)
    start_session_time_train = pd.DataFrame(X_train['time1'].apply(lambda x: 100 * x.year + x.month))
    start_session_time_test = pd.DataFrame(X_test['time1'].apply(lambda x: 100 * x.year + x.month))

    scaler = StandardScaler().fit(start_session_time_train)

    start_session_time_train = scaler.transform(start_session_time_train)
    start_session_time_test = scaler.transform(start_session_time_test)


    # Start hour
    start_hour_train = pd.DataFrame(X_train['time1'].apply(lambda x: x.hour))
    start_hour_test = pd.DataFrame(X_test['time1'].apply(lambda x: x.hour))

    scaler = StandardScaler().fit(start_hour_train)

    start_hour_train = scaler.transform(start_hour_train)
    start_hour_test = scaler.transform(start_hour_test)


    # Start morning
    binary_morning_train = pd.DataFrame(X_train['time1'].apply(lambda x: 1 if x.hour <= 11 else 0))
    binary_morning_test = pd.DataFrame(X_test['time1'].apply(lambda x: 1 if x.hour <= 11 else 0))


    # Domens
    with open(r'data/site_dic.pkl', 'rb') as input_file:
        site_dict = pickle.load(input_file)

    invert_site_dict = {v: k for k, v in site_dict.items()}
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_domens_only = X_train[sites].applymap(lambda x: get_domen(invert_site_dict[x]) if x in invert_site_dict else 'nan')
    X_test_domens_only = X_test[sites].applymap(lambda x: get_domen(invert_site_dict[x]) if x in invert_site_dict else 'nan')

    X_train_domens_only = X_train_domens_only[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)
    X_test_domens_only = X_test_domens_only[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)

    domens_tfidf_vectorizer = CountVectorizer()
    domens_tfidf_vectorizer.fit(X_train_domens_only)

    domens_train_tfidf = domens_tfidf_vectorizer.transform(X_train_domens_only)
    domens_test_tfidf = domens_tfidf_vectorizer.transform(X_test_domens_only)


    # Final concat
    X_train = csr_matrix(hstack([train_tfidf, start_session_time_train, start_hour_train, binary_morning_train, domens_train_tfidf]))
    X_test = csr_matrix(hstack([test_tfidf, start_session_time_test, start_hour_test, binary_morning_test, domens_test_tfidf]))

    return X_train, X_test

def variant_4_preprocessing(X_train, X_test, y_train):
    '''
    TfidfVectorizer + domens CountVectorizer + start session time (yyyymm) + start_hour + morning(12) + start day of week + session duration (seconds) +
    mean time on one site + count Alice top sites(50)
    '''
    # TfidfVectorizer
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_sites_only = X_train[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)
    X_test_sites_only = X_test[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X_train_sites_only)

    train_tfidf = tfidf_vectorizer.transform(X_train_sites_only)
    test_tfidf = tfidf_vectorizer.transform(X_test_sites_only)


    # Domens
    with open(r'data/site_dic.pkl', 'rb') as input_file:
        site_dict = pickle.load(input_file)

    invert_site_dict = {v: k for k, v in site_dict.items()}
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_domens_only = X_train[sites].applymap(lambda x: get_domen(invert_site_dict[x]) if x in invert_site_dict else 'nan')
    X_test_domens_only = X_test[sites].applymap(lambda x: get_domen(invert_site_dict[x]) if x in invert_site_dict else 'nan')

    X_train_domens_only = X_train_domens_only[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)
    X_test_domens_only = X_test_domens_only[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)

    domens_tfidf_vectorizer = CountVectorizer()
    domens_tfidf_vectorizer.fit(X_train_domens_only)

    domens_train_tfidf = domens_tfidf_vectorizer.transform(X_train_domens_only)
    domens_test_tfidf = domens_tfidf_vectorizer.transform(X_test_domens_only)


    # Start session time (yyyymm)
    start_session_time_train = pd.DataFrame(X_train['time1'].apply(lambda x: 100 * x.year + x.month))
    start_session_time_test = pd.DataFrame(X_test['time1'].apply(lambda x: 100 * x.year + x.month))

    scaler_yearmonth = StandardScaler().fit(start_session_time_train)

    start_session_time_train = scaler_yearmonth.transform(start_session_time_train)
    start_session_time_test = scaler_yearmonth.transform(start_session_time_test)


    # Start hour
    start_hour_train = pd.DataFrame(X_train['time1'].apply(lambda x: x.hour))
    start_hour_test = pd.DataFrame(X_test['time1'].apply(lambda x: x.hour))

    scaler_hour = StandardScaler().fit(start_hour_train)

    start_hour_train = scaler_hour.transform(start_hour_train)
    start_hour_test = scaler_hour.transform(start_hour_test)


    # Start morning
    binary_morning_train = pd.DataFrame(X_train['time1'].apply(lambda x: 1 if x.hour <= 12 else 0))
    binary_morning_test = pd.DataFrame(X_test['time1'].apply(lambda x: 1 if x.hour <= 12 else 0))


    # Start day of week
    day_of_week_train = pd.DataFrame(X_train['time1'].apply(lambda x: x.dayofweek))
    day_of_week_test = pd.DataFrame(X_test['time1'].apply(lambda x: x.dayofweek))

    scaler_dayofweek = StandardScaler().fit(day_of_week_train)

    day_of_week_train = scaler_dayofweek.transform(day_of_week_train)
    day_of_week_test = scaler_dayofweek.transform(day_of_week_test)


    # Session duration (seconds)
    times = ['time%s' % i for i in range(1, 11)]
    X_train_times = X_train[times]
    X_test_times = X_test[times]

    duration_train = pd.DataFrame((X_train_times.max(axis=1) - X_train_times.min(axis=1)).dt.total_seconds())
    duration_test = pd.DataFrame((X_test_times.max(axis=1) - X_test_times.min(axis=1)).dt.total_seconds())

    scaler_duration = StandardScaler().fit(duration_train)

    duration_train = scaler_duration.transform(duration_train)
    duration_test = scaler_duration.transform(duration_test)


    # Mean time on one site
    times = ['time%s' % i for i in range(1, 11)]
    X_train_times = X_train[times]
    X_test_times = X_test[times]

    def one_site_mean_duration(x):
        site_times = [datetime for datetime in list(x) if not pd.isnull(datetime)]
        durations = [site_times[i] - site_times[i-1] for i in range(1, len(site_times))]
        durations = list(map(lambda x: x.seconds, durations))

        if (len(durations) > 0):
            return np.mean(durations)

        return 0

    one_site_mean_duration_train = pd.DataFrame(X_train_times.apply(one_site_mean_duration, axis=1))
    one_site_mean_duration_test = pd.DataFrame(X_test_times.apply(one_site_mean_duration, axis=1))

    scaler_mean_duration = StandardScaler().fit(one_site_mean_duration_train)

    one_site_mean_duration_train = scaler_mean_duration.transform(one_site_mean_duration_train)
    one_site_mean_duration_test = scaler_mean_duration.transform(one_site_mean_duration_test)


    # Count Alice top sites(50)
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_sites_alice = X_train.iloc[y_train[y_train == 1].index, :][sites]
    alice_sites = X_train_sites_alice.stack().value_counts()
    alice_top_sites = list(alice_sites.drop(alice_sites.index[0]))[:50]

    def count_alice_top_sites(top_sites, x):
        x_list = list(x)
        x_list = [site for site in x_list if site in top_sites]

        return len(x_list)

    X_train_sites = pd.DataFrame(X_train[sites].apply((lambda x: count_alice_top_sites(alice_top_sites, x)), axis=1))
    X_test_sites = pd.DataFrame(X_test[sites].apply((lambda x: count_alice_top_sites(alice_top_sites, x)), axis=1))

    scaler_top_sites = StandardScaler().fit(X_train_sites)

    X_train_sites = scaler_top_sites.transform(X_train_sites)
    X_test_sites = scaler_top_sites.transform(X_test_sites)


    # Final concat
    X_train = csr_matrix(hstack([train_tfidf, start_session_time_train, start_hour_train, binary_morning_train, domens_train_tfidf, day_of_week_train, duration_train, one_site_mean_duration_train, X_train_sites]))
    X_test = csr_matrix(hstack([test_tfidf, start_session_time_test, start_hour_test, binary_morning_test, domens_test_tfidf, day_of_week_test, duration_test, one_site_mean_duration_test, X_test_sites]))

    return X_train, X_test

def variant_5_preprocessing(X_train, X_test, y_train):
    '''
    Transfer to OHE
    Sites TfidfVectorizer + Domens TfidfVectorizer + Start Hour (OHE) + Start Session Time (yyyy/mm) (OHE/OHE) + Start Part Of Day (OHE) + Start Day Of Week (OHE)
    + Start Is Weekend (OHE) + Session Duration (seconds) + Mean Time On One Site (Seconds) + Count Alice Top Sites(10) + Start Day Of Year (OHE)

    '''
    with open(r'data/site_dic.pkl', 'rb') as input_file:
        site_dict = pickle.load(input_file)

    # Sites
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_sites = X_train[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)
    X_test_sites = X_test[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)

    sites_vectorizer = TfidfVectorizer(max_features=10000, max_df=0.1, ngram_range=(1, 2)).fit(X_train_sites)
    sites_train = sites_vectorizer.transform(X_train_sites)
    sites_test = sites_vectorizer.transform(X_test_sites)


    # Domens
    invert_site_dict = {v: k for k, v in site_dict.items()}
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_domens = X_train[sites].applymap(lambda x: get_domen(invert_site_dict[x]) if x in invert_site_dict else 'nan')
    X_test_domens = X_test[sites].applymap(lambda x: get_domen(invert_site_dict[x]) if x in invert_site_dict else 'nan')

    X_train_domens = X_train_domens[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)
    X_test_domens = X_test_domens[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)

    domens_vectorizer = CountVectorizer(max_df=0.1, min_df=0.02).fit(X_train_domens)
    domens_train = domens_vectorizer.transform(X_train_domens)
    domens_test = domens_vectorizer.transform(X_test_domens)


    # Start Hour (OHE)
    start_hour_train_catseries = X_train['time1'].apply(lambda x: x.hour).astype('category')
    start_hour_test_catseries = X_test['time1'].apply(lambda x: x.hour).astype('category', categories=list(start_hour_train_catseries.cat.categories))

    start_hour_train = pd.get_dummies(start_hour_train_catseries)
    start_hour_test = pd.get_dummies(start_hour_test_catseries)


    # Start Session Time (yyyy/mm) (OHE/OHE)
    start_year_train_catseries = X_train['time1'].apply(lambda x: x.year).astype('category')
    start_year_test_catseries = X_test['time1'].apply(lambda x: x.year).astype('category', categories=list(start_year_train_catseries.cat.categories))

    start_year_train = pd.get_dummies(start_year_train_catseries)
    start_year_test = pd.get_dummies(start_year_test_catseries)


    start_month_train_catseries = X_train['time1'].apply(lambda x: x.month).astype('category')
    start_month_test_catseries = X_test['time1'].apply(lambda x: x.month).astype('category', categories=list(start_month_train_catseries.cat.categories))

    start_month_train = pd.get_dummies(start_month_train_catseries)
    start_month_test = pd.get_dummies(start_month_test_catseries)


    # Start Part Of Day (OHE)
    part_of_day_train_catseries = X_train['time1'].apply(get_part_of_day).astype('category')
    part_of_day_test_catseries = X_test['time1'].apply(get_part_of_day).astype('category', categories=list(part_of_day_train_catseries.cat.categories))

    part_of_day_train = pd.get_dummies(part_of_day_train_catseries)
    part_of_day_test = pd.get_dummies(part_of_day_test_catseries)


    # Start Day Of Week (OHE)
    day_of_week_train_catseries = X_train['time1'].apply(lambda x: x.dayofweek).astype('category')
    day_of_week_test_catseries = X_test['time1'].apply(lambda x: x.dayofweek).astype('category', categories=list(day_of_week_train_catseries.cat.categories))

    day_of_week_train = pd.get_dummies(day_of_week_train_catseries)
    day_of_week_test = pd.get_dummies(day_of_week_test_catseries)


    # # Start Is Weekend (OHE)
    # is_weekend_train_catseries = X_train['time1'].apply(lambda x: 1 if x.dayofweek >= 5 else 0).astype('category')
    # is_weekend_test_catseries = X_test['time1'].apply(lambda x: x.dayofweek).astype('category', categories=list(is_weekend_train_catseries.cat.categories))
    #
    # is_weekend_train = pd.get_dummies(is_weekend_train_catseries)
    # is_weekend_test = pd.get_dummies(is_weekend_test_catseries)


    # Session Duration (Seconds)
    times = ['time%s' % i for i in range(1, 11)]
    X_train_times = X_train[times]
    X_test_times = X_test[times]

    duration_train = pd.DataFrame((X_train_times.max(axis=1) - X_train_times.min(axis=1)).dt.total_seconds())
    duration_test = pd.DataFrame((X_test_times.max(axis=1) - X_test_times.min(axis=1)).dt.total_seconds())

    scaler_duration = StandardScaler().fit(duration_train)

    duration_train = scaler_duration.transform(duration_train)
    duration_test = scaler_duration.transform(duration_test)


    # Mean Time On One Site (Seconds)
    times = ['time%s' % i for i in range(1, 11)]
    X_train_times = X_train[times]
    X_test_times = X_test[times]

    one_site_mean_duration_train = pd.DataFrame(X_train_times.apply(one_site_mean_duration, axis=1))
    one_site_mean_duration_test = pd.DataFrame(X_test_times.apply(one_site_mean_duration, axis=1))

    scaler_mean_duration = StandardScaler().fit(one_site_mean_duration_train)

    one_site_mean_duration_train = scaler_mean_duration.transform(one_site_mean_duration_train)
    one_site_mean_duration_test = scaler_mean_duration.transform(one_site_mean_duration_test)


    # Count Alice Top Sites(50)
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_sites_alice = X_train.iloc[y_train[y_train == 1].index, :][sites]
    alice_sites = X_train_sites_alice.stack().value_counts()
    alice_top_sites = list(alice_sites.drop(alice_sites.index[0]))[:10]

    alice_sites_train = pd.DataFrame(X_train[sites].apply((lambda x: count_alice_top_sites(alice_top_sites, x)), axis=1))
    alice_sites_test = pd.DataFrame(X_test[sites].apply((lambda x: count_alice_top_sites(alice_top_sites, x)), axis=1))

    scaler_top_sites = StandardScaler().fit(alice_sites_train)

    alice_sites_train = scaler_top_sites.transform(alice_sites_train)
    alice_sites_test = scaler_top_sites.transform(alice_sites_test)


    # Start Day Of Year (OHE)
    day_of_year_train_catseries = X_train['time1'].apply(lambda x: x.dayofyear).astype('category')
    day_of_year_test_catseries = X_test['time1'].apply(lambda x: x.dayofyear).astype('category', categories=list(day_of_year_train_catseries.cat.categories))

    day_of_year_train = pd.get_dummies(day_of_year_train_catseries)
    day_of_year_test = pd.get_dummies(day_of_year_test_catseries)


    # Final concat
    X_train = csr_matrix(hstack([sites_train, domens_train, start_hour_train, start_year_train, start_month_train, part_of_day_train, day_of_week_train, duration_train, one_site_mean_duration_train, alice_sites_train, day_of_year_train]))
    X_test = csr_matrix(hstack([sites_test, domens_test, start_hour_test, start_year_test, start_month_test, part_of_day_test, day_of_week_test, duration_test, one_site_mean_duration_test, alice_sites_test, day_of_year_test]))

    return X_train, X_test

def variant_6_preprocessing(X_train, X_test, y_train):
    '''
    Transfer to OHE
    Sites TfidfVectorizer + Domens TfidfVectorizer + Start Hour (OHE) + Start Session Time (yyyy/mm) (OHE/OHE) + Start Part Of Day (OHE) + Start Day Of Week (OHE)
    + Start Is Weekend (OHE) + Session Duration (seconds) + Mean Time On One Site (Seconds) + Count Alice Top Sites(10) + Start Day Of Year (OHE)

    '''
    with open(r'data/site_dic.pkl', 'rb') as input_file:
        site_dict = pickle.load(input_file)

    # Sites
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_sites = X_train[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)
    X_test_sites = X_test[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)

    sites_vectorizer = TfidfVectorizer(max_features=10000, max_df=0.1, ngram_range=(1, 2)).fit(X_train_sites)
    sites_train = sites_vectorizer.transform(X_train_sites)
    sites_test = sites_vectorizer.transform(X_test_sites)


    # Domens
    invert_site_dict = {v: k for k, v in site_dict.items()}
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_domens = X_train[sites].applymap(lambda x: get_domen(invert_site_dict[x]) if x in invert_site_dict else 'nan')
    X_test_domens = X_test[sites].applymap(lambda x: get_domen(invert_site_dict[x]) if x in invert_site_dict else 'nan')

    X_train_domens = X_train_domens[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)
    X_test_domens = X_test_domens[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)

    domens_vectorizer = TfidfVectorizer(max_df=0.1, min_df=0.02).fit(X_train_domens)
    domens_train = domens_vectorizer.transform(X_train_domens)
    domens_test = domens_vectorizer.transform(X_test_domens)


    # Start Hour (OHE)
    start_hour_train_catseries = X_train['time1'].apply(lambda x: x.hour).astype('category')
    start_hour_test_catseries = X_test['time1'].apply(lambda x: x.hour).astype('category', categories=list(start_hour_train_catseries.cat.categories))

    start_hour_train = pd.get_dummies(start_hour_train_catseries)
    start_hour_test = pd.get_dummies(start_hour_test_catseries)


    # Start Session Time (yyyy/mm) (OHE/OHE)
    start_year_train_catseries = X_train['time1'].apply(lambda x: x.year).astype('category')
    start_year_test_catseries = X_test['time1'].apply(lambda x: x.year).astype('category', categories=list(start_year_train_catseries.cat.categories))

    start_year_train = pd.get_dummies(start_year_train_catseries)
    start_year_test = pd.get_dummies(start_year_test_catseries)


    start_month_train_catseries = X_train['time1'].apply(lambda x: x.month).astype('category')
    start_month_test_catseries = X_test['time1'].apply(lambda x: x.month).astype('category', categories=list(start_month_train_catseries.cat.categories))

    start_month_train = pd.get_dummies(start_month_train_catseries)
    start_month_test = pd.get_dummies(start_month_test_catseries)


    # Start Part Of Day (OHE)
    part_of_day_train_catseries = X_train['time1'].apply(get_part_of_day).astype('category')
    part_of_day_test_catseries = X_test['time1'].apply(get_part_of_day).astype('category', categories=list(part_of_day_train_catseries.cat.categories))

    part_of_day_train = pd.get_dummies(part_of_day_train_catseries)
    part_of_day_test = pd.get_dummies(part_of_day_test_catseries)


    # Start Day Of Week (OHE)
    day_of_week_train_catseries = X_train['time1'].apply(lambda x: x.dayofweek).astype('category')
    day_of_week_test_catseries = X_test['time1'].apply(lambda x: x.dayofweek).astype('category', categories=list(day_of_week_train_catseries.cat.categories))

    day_of_week_train = pd.get_dummies(day_of_week_train_catseries)
    day_of_week_test = pd.get_dummies(day_of_week_test_catseries)


    # Start Is Weekend (OHE)
    is_weekend_train_catseries = X_train['time1'].apply(lambda x: 1 if x.dayofweek >= 5 else 0).astype('category')
    is_weekend_test_catseries = X_test['time1'].apply(lambda x: x.dayofweek).astype('category', categories=list(is_weekend_train_catseries.cat.categories))

    is_weekend_train = pd.get_dummies(is_weekend_train_catseries)
    is_weekend_test = pd.get_dummies(is_weekend_test_catseries)


    # Session Duration (Seconds)
    times = ['time%s' % i for i in range(1, 11)]
    X_train_times = X_train[times]
    X_test_times = X_test[times]

    duration_train = pd.DataFrame((X_train_times.max(axis=1) - X_train_times.min(axis=1)).dt.total_seconds())
    duration_test = pd.DataFrame((X_test_times.max(axis=1) - X_test_times.min(axis=1)).dt.total_seconds())

    scaler_duration = StandardScaler().fit(duration_train)

    duration_train = scaler_duration.transform(duration_train)
    duration_test = scaler_duration.transform(duration_test)


    # Mean Time On One Site (Seconds)
    times = ['time%s' % i for i in range(1, 11)]
    X_train_times = X_train[times]
    X_test_times = X_test[times]

    one_site_mean_duration_train = pd.DataFrame(X_train_times.apply(one_site_mean_duration, axis=1))
    one_site_mean_duration_test = pd.DataFrame(X_test_times.apply(one_site_mean_duration, axis=1))

    scaler_mean_duration = StandardScaler().fit(one_site_mean_duration_train)

    one_site_mean_duration_train = scaler_mean_duration.transform(one_site_mean_duration_train)
    one_site_mean_duration_test = scaler_mean_duration.transform(one_site_mean_duration_test)


    # Count Alice Top Sites(50)
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_sites_alice = X_train.iloc[y_train[y_train == 1].index, :][sites]
    alice_sites = X_train_sites_alice.stack().value_counts()
    alice_top_sites = list(alice_sites.drop(alice_sites.index[0]))[:10]

    X_train_sites = pd.DataFrame(X_train[sites].apply((lambda x: count_alice_top_sites(alice_top_sites, x)), axis=1))
    X_test_sites = pd.DataFrame(X_test[sites].apply((lambda x: count_alice_top_sites(alice_top_sites, x)), axis=1))

    scaler_top_sites = StandardScaler().fit(X_train_sites)

    X_train_sites = scaler_top_sites.transform(X_train_sites)
    X_test_sites = scaler_top_sites.transform(X_test_sites)


    # Start Day Of Year (OHE)
    day_of_year_train_catseries = X_train['time1'].apply(lambda x: x.dayofyear).astype('category')
    day_of_year_test_catseries = X_test['time1'].apply(lambda x: x.dayofyear).astype('category', categories=list(day_of_year_train_catseries.cat.categories))

    day_of_year_train = pd.get_dummies(day_of_year_train_catseries)
    day_of_year_test = pd.get_dummies(day_of_year_test_catseries)


    # Prefinal concat
    X_train_sparse = csr_matrix(hstack([sites_train, domens_train, start_hour_train, start_year_train, start_month_train, part_of_day_train, day_of_week_train, is_weekend_train, duration_train, one_site_mean_duration_train, X_train_sites, day_of_year_train]))
    X_test_sparse = csr_matrix(hstack([sites_test, domens_test, start_hour_test, start_year_test, start_month_test, part_of_day_test, day_of_week_test, is_weekend_test, duration_test, one_site_mean_duration_test, X_test_sites, day_of_year_test]))


    # kNN Feature
    svd = TruncatedSVD(n_components=30, n_iter=7, random_state=17).fit(X_train_sparse)
    X_train_reduction = svd.transform(X_train_sparse)
    X_test_reduction = svd.transform(X_test_sparse)

    knn = KNeighborsClassifier(n_neighbors=699, n_jobs=-1)
    knn.fit(X_train_reduction, y_train)
    knn_predicitons_train = (knn.predict_proba(X_train_reduction)[:, 1]).reshape((-1, 1))
    knn_predicitons_test = (knn.predict_proba(X_test_reduction)[:, 1]).reshape((-1, 1))


    # # RF Leaves Feature
    # rf = RandomForestClassifier(max_depth=200, max_features=0.4, min_samples_leaf=410)
    # rf.fit(X_train_reduction, y_train)
    #
    # common_leaves_ids_train = (rf.predict_proba(X_train_reduction)[:, 1]).reshape((-1, 1))
    # common_leaves_ids_test = (rf.predict_proba(X_test_reduction)[:, 1]).reshape((-1, 1))


    # Final concat
    X_train = csr_matrix(hstack([X_train_sparse, knn_predicitons_train]))
    X_test = csr_matrix(hstack([X_test_sparse, knn_predicitons_test]))


    return X_train, X_test

def variant_7_preprocessing(X_train, X_test, y_train):
    '''
    Sites TfidfVectorizer + Domens CountVectorizer + Start Hour (OHE) + Start Session Time (yyyy/mm) (OHE/OHE) + Start Part Of Day (OHE) + Start Day Of Week (OHE)
    + Session Duration (seconds) + Mean Time On One Site (Seconds) + Count Alice Top Sites(10) + Start Day Of Year (OHE)

    '''
    with open(r'data/site_dic.pkl', 'rb') as input_file:
        site_dict = pickle.load(input_file)

    # Sites
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_sites = X_train[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)
    X_test_sites = X_test[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)

    sites_vectorizer = TfidfVectorizer(max_features=10000, max_df=0.1, ngram_range=(1, 2)).fit(X_train_sites)
    sites_train = sites_vectorizer.transform(X_train_sites)
    sites_test = sites_vectorizer.transform(X_test_sites)


    # Domens
    invert_site_dict = {v: k for k, v in site_dict.items()}
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_domens = X_train[sites].applymap(lambda x: get_domen(invert_site_dict[x]) if x in invert_site_dict else 'nan')
    X_test_domens = X_test[sites].applymap(lambda x: get_domen(invert_site_dict[x]) if x in invert_site_dict else 'nan')

    X_train_domens = X_train_domens[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)
    X_test_domens = X_test_domens[sites].apply(lambda x: ' '.join(map(str, x.values)), axis=1)

    domens_vectorizer = CountVectorizer(max_df=0.1, min_df=0.02).fit(X_train_domens)
    domens_train = domens_vectorizer.transform(X_train_domens)
    domens_test = domens_vectorizer.transform(X_test_domens)


    # Start Hour (OHE)
    start_hour_train_catseries = X_train['time1'].apply(lambda x: x.hour).astype('category')
    start_hour_test_catseries = X_test['time1'].apply(lambda x: x.hour).astype('category', categories=list(start_hour_train_catseries.cat.categories))

    start_hour_train = pd.get_dummies(start_hour_train_catseries)
    start_hour_test = pd.get_dummies(start_hour_test_catseries)


    # Start Session Time (yyyy/mm) (OHE/OHE)
    start_year_train_catseries = X_train['time1'].apply(lambda x: x.year).astype('category')
    start_year_test_catseries = X_test['time1'].apply(lambda x: x.year).astype('category', categories=list(start_year_train_catseries.cat.categories))

    start_year_train = pd.get_dummies(start_year_train_catseries)
    start_year_test = pd.get_dummies(start_year_test_catseries)


    start_month_train_catseries = X_train['time1'].apply(lambda x: x.month).astype('category')
    start_month_test_catseries = X_test['time1'].apply(lambda x: x.month).astype('category', categories=list(start_month_train_catseries.cat.categories))

    start_month_train = pd.get_dummies(start_month_train_catseries)
    start_month_test = pd.get_dummies(start_month_test_catseries)


    # Start Part Of Day (OHE)
    part_of_day_train_catseries = X_train['time1'].apply(get_part_of_day).astype('category')
    part_of_day_test_catseries = X_test['time1'].apply(get_part_of_day).astype('category', categories=list(part_of_day_train_catseries.cat.categories))

    part_of_day_train = pd.get_dummies(part_of_day_train_catseries)
    part_of_day_test = pd.get_dummies(part_of_day_test_catseries)


    # Start Day Of Week (OHE)
    day_of_week_train_catseries = X_train['time1'].apply(lambda x: x.dayofweek).astype('category')
    day_of_week_test_catseries = X_test['time1'].apply(lambda x: x.dayofweek).astype('category', categories=list(day_of_week_train_catseries.cat.categories))

    day_of_week_train = pd.get_dummies(day_of_week_train_catseries)
    day_of_week_test = pd.get_dummies(day_of_week_test_catseries)


    # Session Duration (Seconds)
    times = ['time%s' % i for i in range(1, 11)]
    X_train_times = X_train[times]
    X_test_times = X_test[times]

    duration_train = pd.DataFrame((X_train_times.max(axis=1) - X_train_times.min(axis=1)).dt.total_seconds())
    duration_test = pd.DataFrame((X_test_times.max(axis=1) - X_test_times.min(axis=1)).dt.total_seconds())

    scaler_duration = StandardScaler().fit(duration_train)

    duration_train = scaler_duration.transform(duration_train)
    duration_test = scaler_duration.transform(duration_test)


    # Mean Time On One Site (Seconds)
    times = ['time%s' % i for i in range(1, 11)]
    X_train_times = X_train[times]
    X_test_times = X_test[times]

    one_site_mean_duration_train = pd.DataFrame(X_train_times.apply(one_site_mean_duration, axis=1))
    one_site_mean_duration_test = pd.DataFrame(X_test_times.apply(one_site_mean_duration, axis=1))

    scaler_mean_duration = StandardScaler().fit(one_site_mean_duration_train)

    one_site_mean_duration_train = scaler_mean_duration.transform(one_site_mean_duration_train)
    one_site_mean_duration_test = scaler_mean_duration.transform(one_site_mean_duration_test)


    # Count Alice Top Sites(50)
    sites = ['site%s' % i for i in range(1, 11)]
    X_train_sites_alice = X_train.iloc[y_train[y_train == 1].index, :][sites]
    alice_sites = X_train_sites_alice.stack().value_counts()
    alice_top_sites = list(alice_sites.drop(alice_sites.index[0]))[:10]

    alice_sites_train = pd.DataFrame(X_train[sites].apply((lambda x: count_alice_top_sites(alice_top_sites, x)), axis=1))
    alice_sites_test = pd.DataFrame(X_test[sites].apply((lambda x: count_alice_top_sites(alice_top_sites, x)), axis=1))

    scaler_top_sites = StandardScaler().fit(alice_sites_train)

    alice_sites_train = scaler_top_sites.transform(alice_sites_train)
    alice_sites_test = scaler_top_sites.transform(alice_sites_test)


    # Start Day Of Year (OHE)
    day_of_year_train_catseries = X_train['time1'].apply(lambda x: x.dayofyear).astype('category')
    day_of_year_test_catseries = X_test['time1'].apply(lambda x: x.dayofyear).astype('category', categories=list(day_of_year_train_catseries.cat.categories))

    day_of_year_train = pd.get_dummies(day_of_year_train_catseries)
    day_of_year_test = pd.get_dummies(day_of_year_test_catseries)


    # Prefinal concat
    X_train = csr_matrix(hstack([sites_train, domens_train, start_hour_train, start_year_train, start_month_train, part_of_day_train, day_of_week_train, duration_train, one_site_mean_duration_train, alice_sites_train, day_of_year_train]))
    X_test = csr_matrix(hstack([sites_test, domens_test, start_hour_test, start_year_test, start_month_test, part_of_day_test, day_of_week_test, duration_test, one_site_mean_duration_test, alice_sites_test, day_of_year_test]))


    return X_train, X_test









