#!/usr/bin/env python
# coding: utf-8

# In[8]:


pip install tensorflow


# In[10]:


# Imported Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time



# In[11]:


# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections




# In[12]:


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")




# In[50]:


df = pd.read_csv('C:\creditcard.csv')
df.head()


# In[14]:


df.describe()


# In[56]:


# Verificar valores nulos
df.isnull().sum().max()


# In[58]:


#Identificar Colunas
df.columns


# In[59]:


#Classificacao de Fraude e Não Fraude
print('Nao Fraude', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Fraude', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')


# In[60]:


colors = ["#0880DF", "#DF0050"]

sns.countplot(x='Class', data=df, palette=colors)
plt.title('Class Distributions \n (0: Nao Fraude || 1: Fraude)', fontsize=12)


# In[55]:


ig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = df['Amount'].values
time_val = df['Time'].values

sns.distplot(amount_val, ax=ax[0], color='g')
ax[0].set_title('Distribucao da Transacao por Amount', fontsize=12)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribucao da Transacao por Time', fontsize=12)
ax[1].set_xlim([min(time_val), max(time_val)])



plt.show()


# In[47]:


#Selecao da de Subamorstra para auxilio no equilíbrio do modelo  reducao de outliers
from sklearn.preprocessing import StandardScaler, RobustScaler


# In[64]:


df1 =df
df1.head()


# In[65]:


std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df1['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df1['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df1.drop(['Time','Amount'], axis=1, inplace=True)


# In[66]:


df1.head()


# In[80]:


#Separando os dados do modelo
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

print('Nao Fraude', round(df1['Class'].value_counts()[0]/len(df1) * 100,2), '% df')
print('Fraude', round(df1['Class'].value_counts()[1]/len(df1) * 100,2), '% df')

X = df1.drop('Class', axis=1)
y = df1['Class']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    print("Treino:", train_index, "Teste:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]


# In[81]:


# Criação das Matrizes
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values


# In[82]:


#Verificar Distribuição
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

print('Distribuicoes treino: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))


# In[88]:


##Contar casos classificados como fraude e não fraude
counts = df['Class'].value_counts()
print(counts)



# In[91]:


#Balanceamento dos Dados

#Disposiçao dos dados de forma misturada na subamostra

df1=df1.sample(frac=1)

#alocação dos casos conforme as 492 ocorrencias fraudulentas 

fraude_df1 = df1.loc[df1['Class'] == 1]
naofraude_df1 = df1.loc[df1['Class'] == 0][:492]

normal_df1 = pd.concat([fraude_df1, naofraude_df1])

#randomizar os dados nas linhas da subamostra
ndf1 = normal_df1.sample(frac=1, random_state=20)

ndf1.head()



# In[93]:


#Distribuição e Correlação da Amostra pos Balanceamento

print('Distribuicao das classes na subamostra em equilibrio')
print(ndf1['Class'].value_counts()/len(ndf1))



sns.countplot(x='Class', data=ndf1, palette=colors)
plt.title('Distribucao Classes Igual', fontsize=12)
plt.show()


# In[96]:


#Matriz de Correlaçao

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,18))

# Entire DataFrame
corr = df1.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':18}, ax=ax1)
ax1.set_title("Matriz Desbalanceada \ n comparar ", fontsize=12)


sub_sample_corr = ndf1.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':18}, ax=ax2)
ax2.set_title('Matriz de Correlacao da Amostra \n utilizar', fontsize=12)
plt.show()


# In[97]:


#Box Plot

f, axes = plt.subplots(ncols=4, figsize=(18,4))

#Correlacoes Negativas

sns.boxplot(x="Class", y="V17", data=ndf1, palette=colors, ax=axes[0])
axes[0].set_title('V17 X Correlacao Negativa')

sns.boxplot(x="Class", y="V14", data=ndf1, palette=colors, ax=axes[1])
axes[1].set_title('V14 X Correlacao Negativa')


sns.boxplot(x="Class", y="V12", data=ndf1, palette=colors, ax=axes[2])
axes[2].set_title('V12 X Correlacao Negativa')


sns.boxplot(x="Class", y="V10", data=ndf1, palette=colors, ax=axes[3])
axes[3].set_title('V10 X Correlacao Negativa')

plt.show()


# In[99]:


#Correlacoes Positivas

f, axes = plt.subplots(ncols=4, figsize=(18,4))

# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)
sns.boxplot(x="Class", y="V11", data=ndf1, palette=colors, ax=axes[0])
axes[0].set_title('V11 X Correlacao Positiva')

sns.boxplot(x="Class", y="V4", data=ndf1, palette=colors, ax=axes[1])
axes[1].set_title('V4 X Correlacao Positiva')


sns.boxplot(x="Class", y="V2", data=ndf1, palette=colors, ax=axes[2])
axes[2].set_title('V2 X Correlacao Positiva')


sns.boxplot(x="Class", y="V19", data=ndf1, palette=colors, ax=axes[3])
axes[3].set_title('V19 X Correlacao Positiva')

plt.show()


# In[101]:


#Detecção e Tratamento dos Outliers ( a partir da dif 3° - 2° quartil(25 e 75 percentil) )

from scipy.stats import norm

f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))

v14_dfraude = ndf1['V14'].loc[ndf1['Class'] == 1].values
sns.distplot(v14_dfraude,ax=ax1, fit=norm, color='#fb61d4')
ax1.set_title('V14 Distribucao \n Fraude', fontsize=14)

v12_dfraude = ndf1['V12'].loc[ndf1['Class'] == 1].values
sns.distplot(v12_dfraude,ax=ax2, fit=norm, color='#fbd561')
ax2.set_title('V12 Distribucao \n Fraude', fontsize=14)


v10_dfraude = ndf1['V10'].loc[ndf1['Class'] == 1].values
sns.distplot(v10_dfraude,ax=ax3, fit=norm, color='#61d4fb')
ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)

plt.show()


# In[103]:


#Retirada dos Outliers

#Coluna V14

v14_fraude = ndf1['V14'].loc[ndf1['Class'] == 1].values
q25, q75 = np.percentile(v14_fraude, 25), np.percentile(v14_fraude, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v14_dq = q75 - q25
print('dq : {}'.format(v14_dq))

v14_cut_off = v14_dq * 1.5
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
print('Cut Off: {}'.format(v14_cut_off))
print('V14 Lower: {}'.format(v14_lower))
print('V14 Upper: {}'.format(v14_upper))

outliers = [x for x in v14_fraude if x < v14_lower or x > v14_upper]
print('V14 Outliers de Fraude: {}'.format(len(outliers)))
print('V10 outliers:{}'.format(outliers))


ndf1 = ndf1.drop(ndf1[(ndf1['V14'] > v14_upper) | (ndf1['V14'] < v14_lower)].index)


#Coluna V12

v12_fraude = ndf1['V12'].loc[ndf1['Class'] == 1].values
q25, q75 = np.percentile(v12_fraude, 25), np.percentile(v12_fraude, 75)
v12_dq = q75 - q25

v12_cut_off = v12_dq * 1.5
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
print('V12 Lower: {}'.format(v12_lower))
print('V12 Upper: {}'.format(v12_upper))
outliers = [x for x in v12_fraude if x < v12_lower or x > v12_upper]
print('V12 outliers: {}'.format(outliers))
print('V12 Outliers Fraude: {}'.format(len(outliers)))
ndf1 = ndf1.drop(ndf1[(ndf1['V12'] > v12_upper) | (ndf1['V12'] < v12_lower)].index)
print('Numero por retirada outlier: {}'.format(len(ndf1)))


#Coluna V10

v10_fraude = ndf1['V10'].loc[ndf1['Class'] == 1].values
q25, q75 = np.percentile(v10_fraude, 25), np.percentile(v10_fraude, 75)
v10_dq = q75 - q25

v10_cut_off = v10_dq * 1.5
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
print('V10 Lower: {}'.format(v10_lower))
print('V10 Upper: {}'.format(v10_upper))
outliers = [x for x in v10_fraude if x < v10_lower or x > v10_upper]
print('V10 outliers: {}'.format(outliers))
print(' V10 Outliers Fraude: {}'.format(len(outliers)))
ndf1 = ndf1.drop(ndf1[(ndf1['V10'] > v10_upper) | (ndf1['V10'] < v10_lower)].index)
print('Numero por retirada outlier: {}'.format(len(ndf1)))


# In[109]:


#Boxplot

f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,4))

colors = ['#fb8861', '#61d4fb']


#V14

sns.boxplot(x="Class", y="V14", data=ndf1,ax=ax1, palette=colors)
ax1.set_title("V14 \n Reducao outliers", fontsize=12)
ax1.annotate('Poucos Extremos \n outliers', xy=(0.98, -17.5), xytext=(0, -12),
            fontsize=14)
#V12
sns.boxplot(x="Class", y="V12", data=ndf1, ax=ax2, palette=colors)
ax2.set_title("V12 \n Reducao outliers", fontsize=14)
ax2.annotate('Poucos Extremos \n outliers', xy=(0.98, -17.3), xytext=(0, -12),
            fontsize=14)

#V10
sns.boxplot(x="Class", y="V10", data=ndf1, ax=ax3, palette=colors)
ax3.set_title("V10 Feature\n Reducao outliers", fontsize=14)
ax3.annotate('Poucos Extremos \n outliers', xy=(0.95, -16.5), xytext=(0, -12),
            fontsize=14)

plt.show()


# In[110]:


#Clusterizar

X = ndf1.drop('Class', axis=1)
y = ndf1['Class']


# T-SNE -(reduçao de dimensoes dos dados)
t0 = time.time()
X_reduced_tsne = TSNE(n_components=2, random_state=20).fit_transform(X.values)
t1 = time.time()
print("T-SNE {:.2} s".format(t1 - t0))

# PCA
t0 = time.time()
X_reduced_pca = PCA(n_components=2, random_state=20).fit_transform(X.values)
t1 = time.time()
print("PCA  {:.2} s".format(t1 - t0))

# TruncatedSVD
t0 = time.time()
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=20).fit_transform(X.values)
t1 = time.time()
print("Truncated SVD  {:.2} s".format(t1 - t0))


# In[111]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,4))
# labels = ['Nao Fraud', 'Fraude']
f.suptitle('Clusters por Reducao de Dimensao', fontsize=12)


blue_patch = mpatches.Patch(color='#0a85ff', label='Nao Fraude')
red_patch = mpatches.Patch(color='#ff0a0b', label='Fraude')


# t-SNE scatter plot
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='Nao Fraude', linewidths=2)
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraude', linewidths=2)
ax1.set_title('t-SNE', fontsize=14)

ax1.grid(True)

ax1.legend(handles=[blue_patch, red_patch])


# PCA scatter plot
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='Nao Fraude', linewidths=2)
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraude', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)

ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='Nao Fraude', linewidths=2)
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraude', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)

ax3.legend(handles=[blue_patch, red_patch])

plt.show()


# In[112]:


# Ajustes para Relu

X = ndf1.drop('Class', axis=1)
y = ndf1['Class']


# In[113]:


from sklearn.model_selection import train_test_split


# In[122]:


#Subamostra

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[123]:


#Matrizes

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values


# In[124]:


classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}


# In[119]:


from sklearn.model_selection import cross_val_score


for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")


# In[126]:


# Utilizacao do Grid
from sklearn.model_selection import GridSearchCV


# Logistic Regression 
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)

#ReLU
log_reg = grid_log_reg.best_estimator_

knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)

# KNear - Melhor Estimador
knears_neighbors = grid_knears.best_estimator_

# Vetor de Classificacao
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)

# SVC - melhor estimador
svc = grid_svc.best_estimator_

# Arvore de Decisao
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)

# Arvode de Decisao - Melhor estimator
tree_clf = grid_tree.best_estimator_


# In[127]:


#Overfiting

log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')


knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)
print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')

svc_score = cross_val_score(svc, X_train, y_train, cv=5)
print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')

tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)
print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')


# In[129]:


#Subamostra na validacao cruzada

undersample_X = df1.drop('Class', axis=1)
undersample_y = df1['Class']

for train_index, test_index in sss.split(undersample_X, undersample_y):
    print("Train:", train_index, "Test:", test_index)
    undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]
    undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]
    
undersample_Xtrain = undersample_Xtrain.values
undersample_Xtest = undersample_Xtest.values
undersample_ytrain = undersample_ytrain.values
undersample_ytest = undersample_ytest.values 

undersample_accuracy = []
undersample_precision = []
undersample_recall = []
undersample_f1 = []
undersample_auc = []


# Cross Validating the right way

for train, test in sss.split(undersample_Xtrain, undersample_ytrain):
    undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'), log_reg) # SMOTE happens during Cross Validation not before..
    undersample_model = undersample_pipeline.fit(undersample_Xtrain[train], undersample_ytrain[train])
    undersample_prediction = undersample_model.predict(undersample_Xtrain[test])
    
    undersample_accuracy.append(undersample_pipeline.score(original_Xtrain[test], original_ytrain[test]))
    undersample_precision.append(precision_score(original_ytrain[test], undersample_prediction))
    undersample_recall.append(recall_score(original_ytrain[test], undersample_prediction))
    undersample_f1.append(f1_score(original_ytrain[test], undersample_prediction))
    undersample_auc.append(roc_auc_score(original_ytrain[test], undersample_prediction))


# In[130]:


#Curva de Regressao Logistica
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator1, estimator2, estimator3, estimator4, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(20,14), sharey=True)
    if ylim is not None:
        plt.ylim(*ylim)
    
    # 1 Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
    ax1.set_xlabel('Training size (m)')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend(loc="best")
    
    # 2 Estimator 
    train_sizes, train_scores, test_scores = learning_curve(
        estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax2.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax2.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax2.set_title("Knears Neighbors Learning Curve", fontsize=14)
    ax2.set_xlabel('Training size (m)')
    ax2.set_ylabel('Score')
    ax2.grid(True)
    ax2.legend(loc="best")
    
    # 3 Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax3.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax3.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax3.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax3.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax3.set_title("Support Vector Classifier \n Learning Curve", fontsize=14)
    ax3.set_xlabel('Training size (m)')
    ax3.set_ylabel('Score')
    ax3.grid(True)
    ax3.legend(loc="best")
    
    # 4 Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax4.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax4.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax4.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax4.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax4.set_title("Decision Tree Classifier \n Learning Curve", fontsize=14)
    ax4.set_xlabel('Training size (m)')
    ax4.set_ylabel('Score')
    ax4.grid(True)
    ax4.legend(loc="best")
    return plt


# In[131]:


cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=20)
plot_learning_curve(log_reg, knears_neighbors, svc, tree_clf, X_train, y_train, (0.87, 1.01), cv=cv, n_jobs=4)


# In[132]:


from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
# Create a DataFrame with all the scores and the classifiers names.

log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5,
                             method="decision_function")

knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)

svc_pred = cross_val_predict(svc, X_train, y_train, cv=5,
                             method="decision_function")

tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)


# In[133]:


from sklearn.metrics import roc_auc_score

print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))
print('KNears Neighbors: ', roc_auc_score(y_train, knears_pred))
print('Support Vector Classifier: ', roc_auc_score(y_train, svc_pred))
print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_pred))


# In[138]:


log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)


def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr):
    plt.figure(figsize=(16,8))
    plt.title(' Curva ROC \n Top 4 Classificadores', fontsize=18)
    plt.plot(log_fpr, log_tpr, label='ReLu Score Class: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))
    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors ScoreClass : {:.4f}'.format(roc_auc_score(y_train, knears_pred)))
    plt.plot(svc_fpr, svc_tpr, label='Support Vector Score Class: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))
    plt.plot(tree_fpr, tree_tpr, label='Arvore de Decisao Score Class: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('tx Falso Positivo ', fontsize=16)
    plt.ylabel('tx Identificados Positivo', fontsize=16)
  
    plt.legend()
    
graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr)
plt.show()


# In[139]:


def logistic_roc_curve(log_fpr, log_tpr):
    plt.figure(figsize=(12,8))
    plt.title('ReLu Curva ROC', fontsize=16)
    plt.plot(log_fpr, log_tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('tx Falso Positivo ', fontsize=16)
    plt.ylabel('tx Identificados Positivo', fontsize=16)
    plt.axis([-0.01,1,0,1])
    
    
logistic_roc_curve(log_fpr, log_tpr)
plt.show()


# In[154]:


from sklearn.metrics import precision_recall_curve

precision, recall, threshold = precision_recall_curve(y_train, log_reg_pred)


# In[155]:


from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
y_pred = log_reg.predict(X_train)

# Overfitting Case
print('---' * 45)
print('Overfitting: \n')
print('Recall Score: {:.2f}'.format(recall_score(y_train, y_pred)))
print('Precisao Score: {:.2f}'.format(precision_score(y_train, y_pred)))
print('F1 Score: {:.2f}'.format(f1_score(y_train, y_pred)))
print('Acuracia Score: {:.2f}'.format(accuracy_score(y_train, y_pred)))
print('---' * 45)

# Estimativa
print('---' * 45)
print('Estimativa:\n')
print("Acuracia Score: {:.2f}".format(np.mean(undersample_accuracy)))
print("Precisao Score: {:.2f}".format(np.mean(undersample_precision)))
print("Recall Score: {:.2f}".format(np.mean(undersample_recall)))
print("F1 Score: {:.2f}".format(np.mean(undersample_f1)))
print('---' * 45)


# In[156]:


undersample_y_score = log_reg.decision_function(original_Xtest)


# In[157]:


from sklearn.metrics import average_precision_score

undersample_average_precision = average_precision_score(original_ytest, undersample_y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      undersample_average_precision))


# In[153]:


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,6))

precision, recall, _ = precision_recall_curve(original_ytest, undersample_y_score)

plt.step(recall, precision, color='#004a93', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#48a6ff')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('UnderSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(
          undersample_average_precision), fontsize=16)


# In[165]:


# RELU
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV


print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))
print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))

# List to append the score and then find the average
accuracy_lst = []
precision_lst = []
recall_lst = []
f1_lst = []
auc_lst = []

# Classifier with optimal parameters
# log_reg_sm = grid_log_reg.best_estimator_
log_reg_sm = LogisticRegression()




rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)


# Implementing SMOTE Technique 
# Cross Validating the right way
# Parameters
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
for train, test in sss.split(original_Xtrain, original_ytrain):
    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) # SMOTE happens during Cross Validation not before..
    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])
    best_est = rand_log_reg.best_estimator_
    prediction = best_est.predict(original_Xtrain[test])
    
    accuracy_lst.append(pipeline.score(original_Xtrain[test], original_ytrain[test]))
    precision_lst.append(precision_score(original_ytrain[test], prediction))
    recall_lst.append(recall_score(original_ytrain[test], prediction))
    f1_lst.append(f1_score(original_ytrain[test], prediction))
    auc_lst.append(roc_auc_score(original_ytrain[test], prediction))
    
print('')
print("accuracy: {}".format(np.mean(accuracy_lst)))
print("precision: {}".format(np.mean(precision_lst)))
print("recall: {}".format(np.mean(recall_lst)))
print("f1: {}".format(np.mean(f1_lst)))


# In[166]:


labels = ['Nao Fraude', 'Fraude']
smote_prediction = best_est.predict(original_Xtest)
print(classification_report(original_ytest, smote_prediction, target_names=labels))


# In[169]:


y_score = best_est.decision_function(original_Xtest)


# In[170]:


average_precision = average_precision_score(original_ytest, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))


# In[177]:


from imblearn.over_sampling import SMOTE

# SMOTE Technique (OverSampling) After splitting and Cross Validating
sm = SMOTE(sampling_strategy='minority', random_state=20)
# Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)

oversample = SMOTE()
Xsm_train, ysm_train = oversample.fit_resample(original_Xtrain, original_ytrain)

# This will be the data were we are going to 
#Xsm_train, ysm_train = sm.fit_sample(original_Xtrain, original_ytrain)


# In[178]:


# Logistic Regression
t0 = time.time()
log_reg_sm = grid_log_reg.best_estimator_
log_reg_sm.fit(Xsm_train, ysm_train)
t1 = time.time()
print("Fitting oversample data took :{} sec".format(t1 - t0))


# In[179]:


#Regressao Logistica

from sklearn.metrics import confusion_matrix

# RELU
y_pred_log_reg = log_reg_sm.predict(X_test)

#Outros Modelos
y_pred_knear = knears_neighbors.predict(X_test)
y_pred_svc = svc.predict(X_test)
y_pred_tree = tree_clf.predict(X_test)


log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)
kneighbors_cf = confusion_matrix(y_test, y_pred_knear)
svc_cf = confusion_matrix(y_test, y_pred_svc)
tree_cf = confusion_matrix(y_test, y_pred_tree)

fig, ax = plt.subplots(2, 2,figsize=(22,12))


sns.heatmap(log_reg_cf, ax=ax[0][0], annot=True, cmap=plt.cm.copper)
ax[0, 0].set_title("Logistic Regression \n Confusion Matrix", fontsize=14)
ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(kneighbors_cf, ax=ax[0][1], annot=True, cmap=plt.cm.copper)
ax[0][1].set_title("KNearsNeighbors \n Confusion Matrix", fontsize=14)
ax[0][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[0][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(svc_cf, ax=ax[1][0], annot=True, cmap=plt.cm.copper)
ax[1][0].set_title("Suppor Vector Classifier \n Confusion Matrix", fontsize=14)
ax[1][0].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[1][0].set_yticklabels(['', ''], fontsize=14, rotation=360)


# In[180]:


from sklearn.metrics import classification_report


print('Logistic Regression:')
print(classification_report(y_test, y_pred_log_reg))

print('KNears Neighbors:')
print(classification_report(y_test, y_pred_knear))

print('Support Vector Classifier:')
print(classification_report(y_test, y_pred_svc))

print('Support Vector Classifier:')
print(classification_report(y_test, y_pred_tree))


# In[181]:


# Final Score in the test set of logistic regression
from sklearn.metrics import accuracy_score

# Logistic Regression with Under-Sampling
y_pred = log_reg.predict(X_test)
undersample_score = accuracy_score(y_test, y_pred)



# Logistic Regression with SMOTE Technique (Better accuracy with SMOTE t)
y_pred_sm = best_est.predict(original_Xtest)
oversample_score = accuracy_score(original_ytest, y_pred_sm)


d = {'Technique': ['Random UnderSampling', 'Oversampling (SMOTE)'], 'Score': [undersample_score, oversample_score]}
final_df = pd.DataFrame(data=d)

# Move column
score = final_df['Score']
final_df.drop('Score', axis=1, inplace=True)
final_df.insert(1, 'Score', score)

# Note how high is accuracy score it can be misleading! 
final_df


# In[ ]:


#Rede Neural
# Tentativa a partir de modelo undersample e oversample e utilização do algoritimo de optiização Adam para a probabilidade de fraude ou não.
# Utilização da biblioteca Keras Layers


# In[257]:


import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

n_inputs = X_train.shape[1]

undersample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])


# In[247]:


undersample_model.summary()


# In[266]:


#Funcao de Perda
undersample_model.compile(optimizer= 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[267]:


undersample_model.fit(X_train, y_train, validation_split=0.2, batch_size=25, epochs=20, shuffle=True, verbose=2)


# In[268]:


undersample_predictions = undersample_model.predict(original_Xtest, batch_size=200, verbose=0)


# In[269]:


#undersample_fraud_predictions = undersample_model.predict(original_Xtest, batch_size=200, verbose=0)

undersample_fraud_predictions = (model.predict(original_Xtest) < 0.5).astype("int32")


# In[261]:


import itertools

#Matriz de Confusao
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='matriz Confusao',
                          cmap=plt.cm.Blues):
    """
    Matriz de confusao
    Quando normalizada  `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz confusao normalizada")
    else:
        print('Matriz Confusao, sem normalizacao')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[270]:


import itertools


# Matriz Confusao
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Matriz Confusion',
                          cmap=plt.cm.Blues):
    """
    Matriz de confusao
    Quando normalizada  `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusao Normalizada")
    else:
        print('Matriz Confusao, sem normalizacao')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[262]:


undersample_cm = confusion_matrix(original_ytest, undersample_fraud_predictions)
actual_cm = confusion_matrix(original_ytest, original_ytest)
labels = ['No Fraud', 'Fraud']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(undersample_cm, labels, title="Random UnderSample \n  Matriz Confusao", cmap=plt.cm.Reds)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Matriz Confusao\n com total acuracia)", cmap=plt.cm.Blues)


# In[275]:


#n_inputs = Xsm_train.shape[1]

#oversample_model = Sequential([
 #   Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
 #   Dense(32, activation='relu'),
  #  Dense(2, activation='softmax')
#])


n_inputs = Xsm_train.shape[1]

oversample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='sigmoid')
])


# In[276]:


#Compilação do modelo
oversample_model.compile(optimizer= 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[274]:


oversample_model.fit(Xsm_train, ysm_train, validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=2)


# In[283]:


# list all data in history
history = oversample_model.fit(Xsm_train, ysm_train, validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=2)
print(history.history.keys())


# In[284]:


# summarize history for accuracy
plt.figure(figsize=(12,8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[285]:


# summarize history for loss
plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[278]:


oversample_predictions = oversample_model.predict(original_Xtest, batch_size=200, verbose=0)


# In[279]:


#oversample_fraud_predictions = oversample_model.predict_classes(original_Xtest, batch_size=200, verbose=0)

oversample_fraud_predictions = (model.predict(original_Xtest) > 0.5).astype("int32")


# In[280]:


oversample_smote = confusion_matrix(original_ytest, oversample_fraud_predictions)
actual_cm = confusion_matrix(original_ytest, original_ytest)
labels = ['No Fraud', 'Fraud']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(oversample_smote, labels, title="OverSample (SMOTE) \n Matriz Confusao", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Matriz Confusao \n total acuracia", cmap=plt.cm.Greys)


# In[ ]:




