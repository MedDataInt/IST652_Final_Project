#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
IST652 scripting for data analysi 
Final project by Jie Wang 
09/05/2022 

Project Name: Machine models for Prediction of Heart disease

Backgroud: 
In the United States, heart diseases cause more than 859,000 people deaths each year and $216 billion in health care system cost.
Up TO 6.3% of ER visits are related to chest pain. An urgent question in these patients is whether they have an acute coronary syndromes (ACS), as any delay in diagnosis and treatment can have a negative impact on their prognosis.
If patients at low risk for ACS could be recognized early in the diagnostic process, it has the potential to reduce patient burden, length of stay at the ED, frequency of hospitalization and costs.   

Data: 
^^^https://archive.ics.uci.edu/
^^^^^^prcessed.switzeland.data; 
^^^^^^processed.hungarian.data; 
^^^^^^processed.cleveland.data; 
^^^^^^processed.va.data
^^^https://pubmed.ncbi.nlm.nih.gov/

Aims: 
^^^Aim 1: To explore general profiles of healthy control vs. heard disease patients.  
^^^Aim 2: To explore a prediction model
^^^Aim 3: meta analysis using pubmed data 

Methods: descriptive analysis and explore analysis by Python
"""


# In[2]:


#########################
########################
###### set directory 
get_ipython().run_line_magic('pwd', '')


# In[3]:


get_ipython().run_line_magic('cd', 'C:\\Documents\\Syracuse\\Course 5_Processing_IST652 Scripting for Data Analysis\\Homework\\9. Project 0203 Sep 09\\raw data')
    


# In[4]:


#########################
########################
###### descritive analysis 
##export libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
import os
import yellowbrick
import pickle

from matplotlib.collections import PathCollection
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split# Import train_test_split function
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from yellowbrick.classifier import PrecisionRecallCurve, ROCAUC, ConfusionMatrix
from yellowbrick.style import set_palette
from yellowbrick.model_selection import LearningCurve, FeatureImportances
from yellowbrick.contrib.wrapper import wrap
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from sklearn.datasets import make_classification
from sklearn.inspection import permutation_importance
from matplotlib import pyplot
from pdpbox import pdp


# In[5]:


## libraries setting 
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 100
set_palette('dark')
sns.set_style('whitegrid')


# In[6]:


## color setting 
sns.color_palette('tab10')


# In[7]:


############  
####reading data 
##reading basic information of the data 
read information about the data
with open('heart-disease.names') as f:
    print(f.read())


# In[8]:


### read cleveland data, hungarian data, switzerland data, and va data 
data_cleveland = pd.read_csv('processed.cleveland.data', names =['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target'])
data_hungarian = pd.read_csv('processed.hungarian.data', names =['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target'])
data_switzeland = pd.read_csv('processed.switzerland.data', names =['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target'])
data_va = pd.read_csv('processed.va.data', names =['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target'])


# In[9]:


##show the cleveland data 
data_cleveland


# In[10]:


##show the hungarian data 
data_hungarian


# In[11]:


##show the switzeland data 
data_switzeland


# In[12]:


##show the va data
data_va


# In[13]:


##combine all four data sets
data_all = [data_cleveland, data_hungarian, data_switzeland, data_va]
data_all = pd.concat(data_all)
data_all 


# In[14]:


###################################
#### clean the data set 
### drop the na value record and show the information about the data 
data_new = data_all.dropna()
data_new.info()


# In[15]:


### show the value with ? character 
data_new.eq('?')


# In[16]:


### remove the records with ? character
data = data_new[~data_new.eq('?').any(1)]
data


# In[17]:


#### Print new Dataset info
print('\033[1m'+'.: Dataset Info :.'+'\033[0m')
print('*' * 40)
print('Total Rows:'+'\033[1m', data.shape[0])
print('\033[0m'+'Total Columns:'+'\033[1m', data.shape[1])
print('\033[0m'+'*' * 40)
print('\n')

# --- Print Dataset Detail ---
print('\033[1m'+'.: Dataset Details :.'+'\033[0m')
print('*' * 40)
data.info(memory_usage = False)


# In[18]:


#### fix the data types for the columns before analysis performed
## sex, cp, fbs, restecg, exang, slope, ca, thal should be object 
## age, threstbps,chol, thalach, should be int
## oldpeak should be float

list1 = ['sex', 'cp']
list2 = ['age', 'trestbps', 'chol','thalach']
list3 = ['trestbps','chol','fbs','restecg','thalach','exang','slope']
data[list3] = data[list3].astype(int)
data['thal'] = data['thal'].astype(float)
data['ca'] = data['ca'].astype(float)
data[list1] = data[list1].astype(object)
data[list2] = data[list2].astype(int)
data['oldpeak'] = data['oldpeak'].astype(float)
data['thal'] = data['thal'].astype(int)
data['ca'] = data['ca'].astype(int)
lst=['fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
data[lst] = data[lst].astype(object)
data.info()


# In[19]:


### view data 
data


# In[20]:


#### fix target value
### as mentioned in the data information showed above, target 0 indicates healthy, target 1,2,3,4 indicates heart disease
data['target'] = data['target'].replace([2,3,4],1)


# In[21]:


#### delete the duplicated indix records
data = data[~data.index.duplicated()]
data


# In[37]:


####descriptive stat
data.select_dtypes(exclude='object').describe().T


# In[ ]:


###########################################
############ Aim 1 ###############
###########################################


# In[22]:


############################
############################
##### Descriptive  analysis
##### visulization 
### I. Oldpeak (ST depression) in healthy and heart disease 
fig = sns.FacetGrid(data, hue="target",aspect=4, palette='rocket')
fig.map(sns.kdeplot,'oldpeak',shade= True)
plt.legend(labels=['Healthy' , 'Heart disease'])


# In[23]:


### 2. Oldpeak (ST trestbps) in healthy and heart disease 
fig = sns.FacetGrid(data, hue="target",aspect=4, palette='rocket')
fig.map(sns.kdeplot,'trestbps',shade= True)
plt.legend(labels=['Healthy' , 'Heart disease'])


# In[24]:


### 3. age in healthy and heart disease 
fig = sns.FacetGrid(data, hue="target",aspect=4, palette='rocket')
fig.map(sns.kdeplot,'age',shade= True)
plt.legend(labels=['Healthy' , 'Heart disease'])


# In[25]:


# 4. Serum cholesterol (in mg/dl) (chol) in healthy and heart disease 
fig = sns.FacetGrid(data, hue="target",aspect=4, palette='rocket')
fig.map(sns.kdeplot,'chol',shade= True)
plt.legend(labels=['Healthy' , 'Heart disease'])


# In[26]:


# 4. Maximum heart rate achieved (thalach) in healthy and heart disease 
fig = sns.FacetGrid(data, hue="target",aspect=4, palette='rocket')
fig.map(sns.kdeplot,'thalach',shade= True)
plt.legend(labels=['Healthy' , 'Heart disease'])


# In[27]:


### check correlation between chol and trestbps
sns.jointplot(data=data,
              x='chol',
              y='trestbps',
              kind='scatter',
              cmap='PuBu'
              )


# In[28]:


# check correlation between chol and age
sns.jointplot(data=data,
              x='chol',
              y='age',
              kind='scatter',
              cmap='PuBu'
              )


# In[29]:


# check correlation between thalach and age
sns.jointplot(data=data,
              x='thalach',
              y='age',
              kind='scatter',
              )


# In[35]:


### check disease in no_angina and with angina 
fig = sns.countplot(x = 'exang', data = data, hue = 'target', palette='Blues')
plt.legend(['Healthy', 'Heart disease'])


# In[32]:


### check disease Number of major vessels colored by fluoroscopy  
fig = sns.countplot(x = 'ca', data = data, hue = 'target', palette='BuGn')
plt.legend(['Healthy', 'Heart disease'])


# In[33]:


### check Fasting blood sugar in healthy and disease
fig = sns.countplot(x = 'fbs', data = data, hue = 'target', palette='Reds')
plt.legend(['Healthy', 'Sick'])
fig.set_xticklabels(labels=[ 'low blood sugar','high blood sugar'])


# In[ ]:


###########################################
############ Aim 2 ###############
###########################################


# In[38]:


###########################################
###########################################
############## Prediction models
### prepare data set 
X=data.drop(columns='target')
y=data['target']


# In[39]:


### normalizing
scaler=MinMaxScaler()
X=scaler.fit_transform(X)


# In[40]:


####define the function for learning curve
def plot_LearningCurv(model):
    loglc = LearningCurve(model,  title='Logistic Regression Learning Curve')
    loglc.fit(X_train, y_train)
    loglc.finalize() 


# In[41]:


####define the function for learning curve
def plot_RoC(model):
    logrocauc = ROCAUC(model, classes=['False', 'True'],
    title='Logistic Regression ROC AUC Plot')
    logrocauc.fit(X_train, y_train)
    logrocauc.score(X_test, y_test)
    logrocauc.finalize()
    plt.show()


# In[42]:


X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, shuffle=True)


# In[43]:


### logistic regression 
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
accuracy_score(y_test,y_pred)
plot_RoC(logreg)
plot_LearningCurv(logreg)


# In[44]:


#### KNN
from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier(n_neighbors=4)
knn_model.fit(X_train,y_train)
y_knn_pred=knn_model.predict(X_test)
KNNAcc = accuracy_score(y_knn_pred, y_test)
print('.:. K-Nearest Neighbour Accuracy:'+'\033[1m {:.2f}%'.format(KNNAcc*100)+' .:.')
plot_RoC(knn_model)
plot_LearningCurv(knn_model)


# In[98]:


# SVM
from sklearn.svm import SVC
svm_model=SVC(probability=True)
svm_model.fit(X_train,y_train)
y_svm_pred=svm_model.predict(X_test)
SVMacc = accuracy_score(y_svm_pred, y_test)
print('.:. SVM Neighbour Accuracy:'+'\033[1m {:.2f}%'.format(SVMacc*100)+' .:.')
plot_RoC(svm_model)
plot_LearningCurv(svm_model)


# In[45]:


### Naive Bays
from sklearn.naive_bayes import GaussianNB
NB_model=GaussianNB(var_smoothing=0.08)
NB_model.fit(X_train, y_train)
y_pred_NB=NB_model.predict(X_test)
NBacc = accuracy_score(y_pred_NB, y_test)
print('.:. NB  Accuracy:'+'\033[1m {:.1f}%'.format(NBacc*100)+' .:.')
plot_RoC(NB_model)
plot_LearningCurv(NB_model)


# In[46]:


### Random Classifier
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators=1000, random_state=1, max_leaf_nodes=20, min_samples_split=15)

RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)
RFacc = accuracy_score(y_pred_RF, y_test)
print('.:. RF  Accuracy:'+'\033[1m {:.1f}%'.format(RFacc*100)+' .:.')
plot_RoC(RF_model)
plot_LearningCurv(RF_model)


# In[47]:


print(X_train)


# In[48]:


#### permutation feature importance with knn for classification

# define dataset
# define the model
model = KNeighborsClassifier()
# fit the model
model.fit(X_train, y_train)
# perform permutation importance
results = permutation_importance(model, X_train, y_train, scoring='accuracy')
# get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[50]:


#### fix the plot, and make it clearer 
## check the importance of each feature
names = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia']
importance_value = {'names': names, 'values': importance}
df = pd.DataFrame(data=importance_value)

# --- print feature importance score ---
print('*' * 25)
print('\033[1m'+'.: Importance Feature :.'+'\033[0m')
print('*' * 25)
print(df)


# In[52]:


###print the importance score plot
fig, ax = plt.subplots(figsize =(16, 9))
ax.barh(names, importance)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=26)


# In[273]:


################################
###################### alternative prediction model
dt =data
dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']


# In[275]:


###### reset the name, make the table more explainable
dt['sex'][dt['sex'] == 0] = 'female'
dt['sex'][dt['sex'] == 1] = 'male'

dt['chest_pain_type'][dt['chest_pain_type'] == 1] = 'typical angina'
dt['chest_pain_type'][dt['chest_pain_type'] == 2] = 'atypical angina'
dt['chest_pain_type'][dt['chest_pain_type'] == 3] = 'non-anginal pain'
dt['chest_pain_type'][dt['chest_pain_type'] == 4] = 'asymptomatic'

dt['fasting_blood_sugar'][dt['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
dt['fasting_blood_sugar'][dt['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

dt['rest_ecg'][dt['rest_ecg'] == 0] = 'normal'
dt['rest_ecg'][dt['rest_ecg'] == 1] = 'ST-T wave abnormality'
dt['rest_ecg'][dt['rest_ecg'] == 2] = 'left ventricular hypertrophy'

dt['exercise_induced_angina'][dt['exercise_induced_angina'] == 0] = 'no'
dt['exercise_induced_angina'][dt['exercise_induced_angina'] == 1] = 'yes'

dt['st_slope'][dt['st_slope'] == 1] = 'upsloping'
dt['st_slope'][dt['st_slope'] == 2] = 'flat'
dt['st_slope'][dt['st_slope'] == 3] = 'downsloping'

dt['thalassemia'][dt['thalassemia'] == 1] = 'normal'
dt['thalassemia'][dt['thalassemia'] == 2] = 'fixed defect'
dt['thalassemia'][dt['thalassemia'] == 3] = 'reversable defect'


# In[235]:


######## dummies the table 
dt['num_major_vessels'] = dt['num_major_vessels'].astype('int')
dt = pd.get_dummies(dt, drop_first=True)


# In[237]:


dt.head()


# In[238]:


#### split the data into test and train set
X_train11, X_test11, y_train11, y_test11 = train_test_split(dt.drop('target', 1), dt['target'], test_size = .2, random_state=10)


# In[239]:


#### predict with random forest classification 
model11 = RandomForestClassifier(max_depth=5)
model11.fit(X_train11, y_train11)


# In[240]:


estimator = model11.estimators_[1]
feature_names = [i for i in X_train11.columns]

y_train_str = y_train11.astype('str')
y_train_str[y_train_str == '0'] = 'no disease'
y_train_str[y_train_str == '1'] = 'disease'
y_train_str = y_train_str.values


# In[242]:


y_predict = model11.predict(X_test11)
y_pred_quant = model11.predict_proba(X_test11)[:, 1]
y_pred_bin = model11.predict(X_test11)


# In[251]:


#####################
##########print important keys for the model
#### resting_blood_pressure for disease prediction
base_features = dt.columns.values.tolist()
base_features.remove('target')

feat_name = 'resting_blood_pressure'
pdp_dist = pdp.pdp_isolate(model=model11, dataset=X_test11, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# In[253]:


#### age for disease prediction
feat_name = 'age'
pdp_dist = pdp.pdp_isolate(model=model11, dataset=X_test11, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# In[254]:


## PDP for feature st_depression
feat_name = 'st_depression'
pdp_dist = pdp.pdp_isolate(model=model11, dataset=X_test11, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# In[278]:


## PDP for feature cholesterol
feat_name = 'cholesterol'
pdp_dist = pdp.pdp_isolate(model=model11, dataset=X_test11, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# In[271]:


## PDP for feature resting_blood_pressure
feat_name = 'resting_blood_pressure'
pdp_dist = pdp.pdp_isolate(model=model11, dataset=X_test11, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# In[281]:


## PDP for feature rest_ecg_normal
feat_name = 'rest_ecg_normal'
pdp_dist = pdp.pdp_isolate(model=model11, dataset=X_test11, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# In[262]:


###  interaction between rest_ecg_normal & st_depression
inter1  =  pdp.pdp_interact(model=model11, dataset=X_test11, model_features=base_features, features=['rest_ecg_normal', 'st_depression'])

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['rest_ecg_normal', 'st_depression'], plot_type='contour')
plt.show()

###  interaction between rest_ecg_left ventricular hypertrophy & st_depression
inter1  =  pdp.pdp_interact(model=model11, dataset=X_test11, model_features=base_features, features=['rest_ecg_left ventricular hypertrophy', 'st_depression'])

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['rest_ecg_left ventricular hypertrophy', 'st_depression'], plot_type='contour')
plt.show()


# In[265]:


###  interaction between sex_male & thalassemia_7
inter1  =  pdp.pdp_interact(model=model11, dataset=X_test11, model_features=base_features, features=['sex_male', 'thalassemia_7'])

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['sex_male', 'reversable defect'], plot_type='contour')
plt.show()


# In[267]:


# interaction between sex_male & thalassemia_reversable defect
inter1  =  pdp.pdp_interact(model=model11, dataset=X_test11, model_features=base_features, features=['sex_male', 'thalassemia_reversable defect'])

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['sex_male', 'thalassemia_normal'], plot_type='contour')
plt.show()


# In[277]:


# interaction between age & cholesterol
inter1  =  pdp.pdp_interact(model=model11, dataset=X_test11, model_features=base_features, features=['age', 'cholesterol'])

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['age', 'cholesterol'], plot_type='contour')
plt.show()


# In[279]:


# interaction between resting_blood_pressure & st_depression
inter1  =  pdp.pdp_interact(model=model11, dataset=X_test11, model_features=base_features, features=['resting_blood_pressure', 'st_depression'])

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['resting_blood_pressure', 'st_depression'], plot_type='contour')
plt.show()


# In[280]:


# interaction between resting_blood_pressure & rest_ecg_normal
inter1  =  pdp.pdp_interact(model=model11, dataset=X_test11, model_features=base_features, features=['resting_blood_pressure', 'rest_ecg_normal'])
pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['resting_blood_pressure', 'rest_ecg_normal'], plot_type='contour')
plt.show()


# In[58]:


###########################################
############ Aim 3 ###############
###########################################


# In[126]:


###############################################
##############################################
###################get information from pubmed 
from Bio import Entrez
from xml.etree.ElementTree import iterparse
import csv
import time
import datetime
import urllib.request as libreq
import feedparser
from unidecode import unidecode


# In[53]:


from Bio import Entrez
def search(query):
    Entrez.email = 'jwang326@syr.edu'
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax='20', ## show 20 only, for view purpose 
                            retmode='xml',
                            term=query)
    results = Entrez.read(handle)
    return results
def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'jwang326@syr.edu'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids)
    results = Entrez.read(handle)
    return results
if __name__ == '__main__':
    results = search('thalassemia')
    id_list = results['IdList']
    papers = fetch_details(id_list)
    for i, paper in enumerate(papers['PubmedArticle']):
        print("{}) {}".format(i+1, paper['MedlineCitation']['Article']['ArticleTitle']))


# In[54]:


##########################
###############meta analysis
import PythonMeta as PMA
import os
def showstudies(studies, dtype):
    text= "%-13s %-18s %-20s \n"%('study ID', 'male group', 'female group')
    text += "%-10s % -10s %-10s % -10s % -10s \n"%(' ', 'e1', 'n1','e2','n2')
    for i in range(len(studies)):
        text += "%-10s % -10s %-10s % -10s % -10s \n" %(
        studies[i][4],
        str(studies[i][0]),
        str(studies[i][1]),
        str(studies[i][2]),
        str(studies[i][3])
        )
    return text 


# In[55]:


def showresults(rults):
    text = '%-12s %-8s %-13s %-2s'%('study ID', 'sample size', 'ES[95% CI]', 'Weight(%)\n')
    for i in range(1, len(rults)):
        text +='%-15s %-6d %-4.2f[%.2f,%.2f]\t%6.2f\n' %(
        rults[i][0],
        rults[i][5],
        rults[i][1],
        rults[i][3],
        rults[i][4],
        100*(rults[i][2]/rults[0][2])
        )
    text+= "%-15s %-6d %-4.2f[%.2f %.2f]\t%6d\n"%(
        rults[0][0],
        rults[0][5],
        rults[0][1],
        rults[0][3],
        rults[0][4],
        100
    )
    text += "total %d studys(N=%d)\n"%(len(rults)-1, rults[0][5])
    text +='heterogeneity: Tau\u00b2 =%.3f '%(rults[0][12]) if not rults[0][12] == None else "heterogeneity: "
    text +='Q(Chisquare) =%.2f(p=%s); I\u00b2 = %s\n'%(
        rults[0][7],
        rults[0][8],
        str(round(rults[0][9],2))+"%")
    text += "Overall effect test: z=%.2f, p=%s\n"%(rults[0][10],rults[0][11])
    
    return text


# In[56]:


def main(sample, settings):
    d = PMA.Data()
    m = PMA.Meta()
    f = PMA.Fig()
    
    #should always tell the datatype first!!!
    d.datatype = settings['datatype']#set data type, 'CATE' for binary data or 'CONT' for continuous data
    studies = d.getdata(d.readfile('binary studies.csv')) #load data
#     m.subgroup=d.dubgroup #get data from a data file, see examples of data files
    m.nototal=d.nototal 
    print(showstudies(studies, d.datatype)) #show studies
    m.datatype= d.datatype  #set data type for meta-analysis calculating
    m.models = settings['models']  #set effect models: 'Fixed' or 'Random'
    m.algorithm = settings['algorithm'] #set algorithm, based on datatype and effect size
    m.effect = settings['effect']  #set effect size:RR/OR/RD for binary data; SMD/MD for continuous data
    results = m.meta(studies, nosubgrp =True)  #performing the analysis
    print(m.models + ' ' + m.algorithm + " " + m.effect) #show results table
    print(showresults(results))
    f.size = [10,8]
    f.nototal =False
    f.dpi =200
    f.forest(results).show()   #show forest plot
    f.funnel(results).show()   #show funnel plot


# In[57]:


if __name__ == '__main__':
    d = PMA.Data()
    samp_cont=d.readfile('binary studies.csv')
    settings = {
    'datatype':'CATE',
    'models':'Fixed',
    'algorithm':'MH',
    'effect':'OR'}
    main(samp_cont, settings)


# In[59]:


############End


# In[ ]:




