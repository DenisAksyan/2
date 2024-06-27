import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import csv
import plotly.express as px
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt

#ОБРАБОТКА ДАННЫХ
#=======================================================================
# Выборка стран
#words = ["Afghanistan", "Africa", "Asia", "Europe", "Oceania", "North America", "South America", "Central America & Caribbean"]

# Выборка строк по странам
#with open("C:/1/terrorist-attacks new.csv", "r") as f:
#    reader = csv.reader(f)
#    rows_keep = [row for row in reader if ((row[0] == "Afghanistan") or (row[0] == "Africa") or (row[0] == "Asia") or (row[0] == "Europe") or (row[0] == "North America") or (row[0] == "South America") or (row[0] == "Central America & Caribbean"))]

# Перезапись по выбранным странам
#with open("C:/1/employees.csv", "w", newline="") as wrt:
#    writer = csv.writer(wrt)
#    writer.writerow(["Entity", "Code", "Year", "Terrorist attacks"])
#    for row in rows_keep:
#        writer.writerow(row)

#df = pd.read_csv("C:/1/terrorist-attacks new.csv")
#=======================================================================

#МИРОВАЯ СТАТИСТИКА
#=======================================================================
#Переменная для показа статистики по всем странам
#df_world = df[df['Entity'] == 'World']

#Вывод статистики по всем странам
#fig, ax = plt.subplots(figsize=(10, 5))
#sns.barplot(data=df_world, x = 'Year', y = 'Terrorist attacks', palette = 'Set1')
#plt.title("Terrorist Attacks - World", fontsize = 14)
#plt.xticks(rotation=90, fontsize = 6)
#df11 = df_world
#Метод аналитического выравнивания (Метод наименьших квадратов) - метод меняющий ряд на одну прямую
# для постановления гипотезы о наличии или отстутствии тренда
#(библиотека plotly.express as px)
#fig = px.scatter(df11, x='Year', y='Terrorist attacks', trendline="ols")
#fig.show()
#=======================================================================


#СРАВНЕНИЕ 2 ЧАСТЕЙ МИРА
#=======================================================================
#Статистика по Африке, Азии и Европе
#options = ['Africa', 'Asia', 'Europe']
#cont_df = df.loc[df['Entity'].isin(options)]
#cont_df.head()
#fig, ax = plt.subplots(figsize=(10, 4))
#sns.lineplot(data=cont_df, x="Year", y="Terrorist attacks", hue="Entity")
#plt.title("Terrorist Attacks - Right", fontsize = 18) # Used to display the title and define the size of the title.

#Статистика по Океании, северной, южной и центральной Америки и Карибским островам
#options = ['Oceania', 'North America', 'South America', 'Central America & Caribbean']
#cont_df = df.loc[df['Entity'].isin(options)]
#cont_df.head()
#fig, ax = plt.subplots(figsize=(10, 4))
#sns.lineplot(data=cont_df, x="Year", y="Terrorist attacks", hue="Entity")
#plt.title("Terrorist Attacks - Left", fontsize = 18) # Used to display the title and define the size of the title.
#plt.show()
#=======================================================================

#ВИЗУАЛИЗАЦИЯ ИЗ ИСХОДНОГО КОДА ВЫБОРКИ
#=======================================================================
#df_country = df.dropna()
#df_country = df_country[(df_country['Entity']!='World')]
#fig = px.choropleth(df_country,
#                    locations='Entity', locationmode='country names',
#                    color = 'Terrorist attacks',hover_name="Entity",
#                    animation_frame="Year",
 #                   color_continuous_scale='Viridis_r')
#fig.update_layout(margin={'r':0,'t':0,'l':0,'b':0}, coloraxis_colorbar=dict(
#    title = 'Terrorist attacks',
#    ticks = 'outside',
#    tickvals = [0,100,200,400,600,800,1000,1500,2000,3000],
#    dtick = 12))
#fig.show()
#=======================================================================

#ТЕПЛОВАЯ КАРТА
#=======================================================================
#Для выборки
#df1 = pd.read_csv('C:/1/5.csv')
#df1 = pd.read_csv('C:/1/1cl.csv')
#print(df1)
#=======================================================================
#plt.figure(figsize=(16,9))
#sns.heatmap(df1.corr(), annot=True)
#plt.show()
#=======================================================================




#df2 = pd.read_csv('C:/1/employees-2-_2_.csv')

#corr = df1.corr()                    # 20 by 20 correlation matrix


from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

#dissimilarity = 1 - np.abs(corr)
#hierarchy = linkage(squareform(dissimilarity), method='average')
#print(hierarchy)
#labels = fcluster(hierarchy, 0.5, criterion='distance')
#print(labels)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import statsmodels. formula.api as smf
from sklearn.decomposition import PCA

#df21= pd.read_csv("C:/1/2cll1.csv", delimiter=',', header=None, skiprows=1, names=['South America'])
#df22= pd.read_csv("C:/1/2cll1.csv", delimiter=',', header=None, skiprows=1, names=['Central America & Caribbean'])
#df11 = pd.read_csv('C:/1/1cl.csv')
#df = pd.read_csv("C:/1/55.csv", delimiter=',', header=None, skiprows=1, names=['Afghanistan','Philippines','Central America & Caribbean',	'Europe','North America','South America','Bangladesh','Brazil','Cameroon','Iran','Iraq','Israel','Middle East & North Africa','Nigeria','Pakistan','Sri Lanka','Syria','Thailand','India'])
#df = df.drop(['Central America & Caribbean','Europe','North America','South America','Brazil','Iran','Israel','Sri Lanka'],axis= 1)
#df = df.drop(['Bangladesh', 'Pakistan','Thailand' ],axis= 1)
#print(df)
#a = 'Syria'
#X = df.drop(a,axis= 1)
#y = df[a]
#print(X)
#print(y)
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.3, random_state=101)
#model = LogisticRegression()
#model = LinearRegression()
#model = RandomForestRegressor(max_depth=30, random_state=402)
#model.fit(X_train,y_train)
#predictions = model.predict(X_test)
#print(
#  'mean_squared_error : ', mean_squared_error(y_test, predictions))
#print(
#  'mean_absolute_error : ', mean_absolute_error(y_test, predictions))
#print('Sr = ', sum(y)/len(y))
#print('SrOtklonenie = ', mean_absolute_error(y_test, predictions)/(sum(y)/len(y)))
#print('')


#df = pd.read_csv("C:/1/55.csv", delimiter=',', header=None, skiprows=1, names=['Afghanistan','Philippines','Central America & Caribbean',	'Europe','North America','South America','Bangladesh','Brazil','Cameroon','Iran','Iraq','Israel','Middle East & North Africa','Nigeria','Pakistan','Sri Lanka','Syria','Thailand','India'])
#df = df.drop(['Europe','North America','Bangladesh','Brazil','Cameroon','Iran','Iraq','Israel','Middle East & North Africa','Nigeria','Pakistan','Sri Lanka','Syria','Thailand','India'],axis= 1)
#a = 'South America'
#X = df.drop(a,axis= 1)
#y = df[a]
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.3, random_state=101)
#model = LogisticRegression()
#model = LinearRegression()
#model = RandomForestRegressor(max_depth=3, random_state=2)
#model = PolynomialFeatures()
#model = PCA()
#model.fit(X_train,y_train)
#predictions = model.predict(X_test)
#print(
#  'mean_squared_error : ', mean_squared_error(y_test, predictions))
#print(
#  'mean_absolute_error : ', mean_absolute_error(y_test, predictions))
#print('Sr = ', sum(y)/len(y))
#print('SrOtklonenie = ', mean_absolute_error(y_test, predictions)/(sum(y)/len(y)))

#df = pd.read_csv("C:/1/2531.csv", delimiter=',', header=None, skiprows=1)
df = pd.read_csv("C:/1/253.csv", delimiter=',', header=None, skiprows=1)

#print(df)
#print(df.columns)
a = 0
X = df.drop(a, axis= 1)
y = df[a]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)
#model3 = LogisticRegression()
#model3 = LinearRegression()
model3 = RandomForestRegressor(max_depth=3, random_state=2)
#model3 = PolynomialFeatures()
#model3 = PCA()
model3.fit(X_train, y_train)
predictions = model3.predict(X_test)
print(
  'Сумма всех отклонений : ', mean_squared_error(y_test, predictions))
#print(
#  'Среднее отклонение : ', mean_absolute_error(y_test, predictions))
print('Среднее кол-во событий = ', sum(y)/len(y))
sum_column = df[a].sum()
print('Всего событий - ', sum_column)
print('SrOtklonenie = ', mean_squared_error(y_test, predictions)/sum_column)
print('\n')

df = pd.read_csv("C:/1/254.csv", delimiter=',', header=None, skiprows=1)
#print(df)
#print(df.columns)
a = 0
X = df.drop(a,axis= 1)
y = df[a]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.9, random_state=101)
#model4 = LogisticRegression()
#model4 = LinearRegression()
model4 = RandomForestRegressor(max_depth=60, random_state=4)
#model4 = PolynomialFeatures()
#model4 = PCA()
model4.fit(X_train, y_train)
predictions = model4.predict(X_test)
#predictions = model4.predict(X_test)
print(
  'Сумма всех отклонений : ', mean_squared_error(y_test, predictions))
#print(
#  'Среднее отклонение : ', mean_absolute_error(y_test, predictions))
print('Среднее кол-во событий = ', sum(y)/len(y))
sum_column = df[a].sum()
print('Всего событий - ', sum_column)
print('SrOtklonenie = ', mean_squared_error(y_test, predictions)/sum_column)
#print('SrOtklonenie = ', ((sum(y)/len(y))/mean_absolute_error(y_test, predictions)) * (sum_column / mean_absolute_error(y_test, predictions)))

for i in range(1, 7):
    df = pd.read_csv(fr"C:/1/test/{i}.csv", delimiter=',', header=None, skiprows=1)
    X = df
    predictions = model3.predict(df)
    print('\n№ - ', i)
#    print(
#      'mean_squared_error : ', mean_squared_error(X, predictions))
#    print(
#      'mean_absolute_error : ', mean_absolute_error(X, predictions))
#    sum_column = df[0].sum()
#    print('Sr = ', sum_column/53)
#    print('SrOtklonenie = ', mean_absolute_error(X, predictions)/(sum_column/53))
#    print('\n')

    print(
    'Сумма всех отклонений : ', mean_squared_error(X, predictions))
  #print(
#  'Среднее отклонение : ', mean_absolute_error(y_test, predictions))
    sum_column = df[0].sum()
    print('Среднее кол-во событий = ', sum_column/53)
    print('Всего событий - ', sum_column)
    print('SrOtklonenie = ', mean_squared_error(X, predictions)/sum_column)
    print('\n')