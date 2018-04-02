# coding: utf8

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from tqdm import tqdm_notebook, tqdm

from matplotlib import pyplot as plt

plt.style.use(['seaborn-darkgrid'])
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.family'] = 'DejaVu Sans'

from sklearn import metrics, decomposition
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

RANDOM_STATE = 17

X_train = np.loadtxt("C:/Users/adm/Downloads/ml_course_open/samsung_HAR/samsung_train.txt")
y_train = np.loadtxt("C:/Users/adm/Downloads/ml_course_open/samsung_HAR/samsung_train_labels.txt").astype(int)

X_test = np.loadtxt("C:/Users/adm/Downloads/ml_course_open/samsung_HAR/samsung_test.txt")
y_test = np.loadtxt("C:/Users/adm/Downloads/ml_course_open/samsung_HAR/samsung_test_labels.txt").astype(int)

# Проверим размерности
assert(X_train.shape == (7352, 561) and y_train.shape == (7352,))
assert(X_test.shape == (2947, 561) and y_test.shape == (2947,))


#Для кластеризации нам не нужен вектор ответов, поэтому будем работать с объединением обучающей и тестовой выборок. Объедините X_train с X_test, а y_train – с y_test.
# Ваш код здесь
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)


#Определим число уникальных значений меток целевого класса.

print np.unique(y)

n_classes = np.unique(y).size

#1 - ходьбе
#2 - подъему вверх по лестнице
#3 - спуску по лестнице
#4 - сидению
#5 - стоянию
#6 - лежанию

#Отмасштабируйте выборку с помощью StandardScaler с параметрами по умолчанию.
# Ваш код здесь
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Понижаем размерность с помощью PCA, оставляя столько компонент, сколько нужно для того, чтобы объяснить как минимум 90% дисперсии исходных
# (отмасштабированных) данных. Используйте отмасштабированную выборку и зафиксируйте random_state (константа RANDOM_STATE).
# Ваш код здесь
pca = decomposition.PCA(random_state=17).fit(X_scaled)
plt.figure(figsize=(10,7))
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 100)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axvline(65, c='b')
plt.axhline(0.9, c='r')
plt.show();

X_pca = pca.transform(X_scaled)

#Вопрос 1:
#Какое минимальное число главных компонент нужно выделить, чтобы объяснить 90% дисперсии исходных (отмасштабированных) данных?
#Ответ 65 (в реальности 64)


#Вопрос 2:
#Сколько процентов дисперсии приходится на первую главную компоненту? Округлите до целых процентов.
#Ответ 51%
print np.cumsum(pca.explained_variance_ratio_)[0]  #0.50738221035

#Визуализируйте данные в проекции на первые две главные компоненты.
# Ваш код здесь
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=20, cmap='viridis');
plt.colorbar()
plt.show();
#Два кластера 1 2 3  и  4 5 6
#Ответ: 2 кластера: (ходьба, подъем вверх по лестнице, спуск по лестнице) и (сидение, стояние, лежание)

#Сделайте кластеризацию данных методом KMeans, обучив модель на данных со сниженной за счет PCA размерностью.
#В данном случае мы подскажем, что нужно искать именно 6 кластеров, но в общем случае мы не будем знать, сколько кластеров надо искать.

#Параметры:

#n_clusters = n_classes (число уникальных меток целевого класса)
#n_init = 100
#random_state = RANDOM_STATE (для воспроизводимости результата)

k_means = KMeans(n_clusters=6, n_init=100, random_state=17)
k_means.fit(X_pca[:,:65].reshape(-1, 65))
k_means_pred = k_means.predict(X_pca[:,:65].reshape(-1, 65))

#Визуализируйте данные в проекции на первые две главные компоненты. Раскрасьте точки в соответствии с полученными метками кластеров.

# Ваш код здесь
#cluster_labels = ['Ходьба', 'Подъем вверх по лестнице', 'Спуск по лестнице', 'Сидение', 'Стояние', 'Лежание']
#cluster_labels = ['Walking', 'Climbing up the stairs', 'Descending the stairs', 'Sitting', 'Standing', 'Lying']
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=k_means_pred, s=20,  cmap='viridis');
plt.colorbar()
plt.show();

#Посмотрите на соответствие между метками кластеров и исходными метками классов и на то, какие виды активностей алгоритм KMeans путает.
tab = pd.crosstab(y, k_means_pred, margins=True)
tab.index = ['ходьба', 'подъем вверх по лестнице',
             'спуск по лестнице', 'сидение', 'стояние', 'лежание', 'все']
tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['все']
print tab

#Видим, что каждому классу (т.е. каждой активности) соответствуют несколько кластеров. Давайте посмотрим на максимальную долю объектов в классе, отнесенных к какому-то одному кластеру.
# Это будет простой метрикой, характеризующей, насколько легко класс отделяется от других при кластеризации.

#Пример: если для класса "спуск по лестнице", в котором 1406 объектов, распределение кластеров такое:
#кластер 1 – 900
#кластер 3 – 500
#кластер 6 – 6,
#то такая доля будет 900 / 1406 ≈ 0.64.

#Вопрос 4:
#Какой вид активности отделился от остальных лучше всего в терминах простой метрики, описанной выше?
dolya = []
for j in range(1,len(tab.index)):
    dolya.append(float(tab[j-1:j][['cluster' + str(i + 1) for i in range(6)]].max(axis=1).values[0])/float(tab[j-1:j][['все']].values[0]))
print dolya
print tab.index[np.array(dolya).argmax()]

#Ответ: подъем вверх по лестнице

#Видно, что kMeans не очень хорошо отличает только активности друг от друга. Используйте метод локтя, чтобы выбрать оптимальное количество кластеров.
# Параметры алгоритма и данные используем те же, что раньше, меняем только n_clusters.

# Ваш код здесь
inertia = []
for k in tqdm(range(1, n_classes + 1)):
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=1).fit(X_pca[:, :65].reshape(-1, 65))
    inertia.append(np.sqrt(kmeans.inertia_))

plt.plot(range(1, n_classes + 1), inertia, marker='s');
plt.xlabel('$k$')
plt.ylabel('$J(C_k)$');
plt.show()

#Вопрос 5:
#Какое количество кластеров оптимально выбрать, согласно методу локтя?

#Ответ: 3


#Попробуем еще один метод кластеризации, который описывался в статье – агломеративную кластеризацию.

ag = AgglomerativeClustering(n_clusters=n_classes, linkage='ward').fit(X_pca[:, :65])
#Посчитайте Adjusted Rand Index (sklearn.metrics) для получившегося разбиения на кластеры и для KMeans с параметрами из задания к 4 вопросу.
from sklearn.metrics import adjusted_rand_score, classification_report, confusion_matrix

print adjusted_rand_score(y,ag.labels_) #0.49362763373004886
print adjusted_rand_score(y,k_means_pred) #0.41980700126
#Вопрос 6:
#Отметьте все верные утверждения.

#Варианты:
#Согласно ARI, KMeans справился с кластеризацией хуже, чем Agglomerative Clustering ++
#Для ARI не имеет значения какие именно метки присвоены кластерам, имеет значение только разбиение объектов на кластеры++++
#В случае случайного разбиения на кластеры ARI будет близок к нулю +++

#Можно заметить, что задача не очень хорошо решается именно как задача кластеризации, если выделять несколько кластеров (> 2).
# Давайте теперь решим задачу классификации, вспомнив, что данные у нас размечены.

#Для классификации используйте метод опорных векторов – класс sklearn.svm.LinearSVC.
#Мы в курсе отдельно не рассматривали этот алгоритм, но он очень известен, почитать про него можно, например, в материалах Евгения Соколова – тут.

#Настройте для LinearSVC гиперпараметр C с помощью GridSearchCV.

#Обучите новый StandardScaler на обучающей выборке (со всеми исходными признаками), примените масштабирование к тестовой выборке
#В GridSearchCV укажите cv=3.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svc = LinearSVC(random_state=RANDOM_STATE)
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
gsv = GridSearchCV(svc, cv=3, param_grid=svc_params)
gsv.fit(X_train_scaled, y_train)
best_svc = gsv.best_params_
print gsv.best_params_, gsv.best_score_ #{'C': 0.1} 0.938248095756

#Вопрос 7
#Какое значение гиперпараметра C было выбрано лучшим по итогам кросс-валидации?
#Ответ: 0.1

y_predicted = gsv.predict(X_test_scaled)
tab = pd.crosstab(y_test, y_predicted, margins=True)
tab.index = ['ходьба', 'подъем вверх по лестнице', 'спуск по лестнице',
             'сидение', 'стояние', 'лежание', 'все']
tab.columns = tab.index
tab

#Вопрос 8:
#Какой вид активности SVM определяет хуже всего в терминах точности? Полноты?

#Ответ:
#по точности – подъем вверх по лестнице, по полноте – лежание
#по точности – лежание, по полноте – сидение
#по точности – ходьба, по полноте – ходьба
#по точности – стояние, по полноте – сидение +++

precision = []
for j in range(1,len(tab.index)):
    precision.append(float(tab[j-1:j][['ходьба', 'подъем вверх по лестнице', 'спуск по лестнице','сидение', 'стояние', 'лежание', 'все']].max(axis=1).values[0])/float(tab[j-1:j][['все']].values[0]))
print precision
print tab.index[np.array(precision).argmax()]

print (classification_report(y_test, y_predicted, target_names=['Walking', 'Climbing up the stairs', 'Descending the stairs', 'Sitting', 'Standing', 'Lying', 'all']))
#                            precision    recall  f1-score   support
#Walking                      0.97      1.00      0.98       496
#Climbing up the stairs       0.98      0.97      0.98       471
#Descending the stairs        1.00      0.98      0.99       420
#Sitting                      0.96      0.87      0.91       491
#Standing                     0.88      0.97      0.92       532
#Lying                        1.00      0.98      0.99       537

#print confusion_matrix(y_test, y_predicted)


#Наконец, проделайте то же самое, что в 7 вопросе, только добавив PCA.
#Используйте выборки X_train_scaled и X_test_scaled
#Обучите тот же PCA, что раньше, на отмасшабированной обучающей выборке, примените преобразование к тестовой
#Настройте гиперпараметр C на кросс-валидации по обучающей выборке с PCA-преобразованием. Вы заметите, насколько это проходит быстрее, чем раньше.
pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

svc = LinearSVC(random_state=RANDOM_STATE)
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
gsv = GridSearchCV(svc, cv=3, param_grid=svc_params)
gsv.fit(X_train_pca[:,:65], y_train)
best_svc = gsv.best_params_
print gsv.best_params_, gsv.best_score_ #{'C': 0.1} 0.899619151251


#Вопрос 9:
#Какова разность между лучшим качеством (долей верных ответов) на кросс-валидации в случае всех 561 исходных признаков и во втором случае, когда применялся метод главных компонент? Округлите до целых процентов.

#Варианты:
#Качество одинаковое
#2%
#4%  ++++
#10%
#20%
#Вопрос 10:
#Выберите все верные утверждения:

#Варианты:

#Метод главных компонент в данном случае позволил уменьшить время обучения модели, при этом качество (доля верных ответов на кросс-валидации) очень пострадало, более чем на 10%
#PCA можно использовать для визуализации данных, однако для этой задачи есть и лучше подходящие методы, например, tSNE. Зато PCA имеет меньшую вычислительную сложность ++
#PCA строит линейные комбинации исходных признаков, и в некоторых задачах они могут плохо интерпретироваться человеком  +++