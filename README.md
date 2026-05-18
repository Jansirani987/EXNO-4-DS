# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set. 
STEP 5:Save the data to the file.
  
# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data

```
<img width="1663" height="727" alt="Screenshot 2026-05-19 013120" src="https://github.com/user-attachments/assets/766dcbdb-0cd9-427e-a8d4-90292fc0c7c3" />

```
data.isnull().sum()

```

<img width="534" height="636" alt="Screenshot 2026-05-19 013424" src="https://github.com/user-attachments/assets/be99f6b3-68a1-4421-b645-35a7a2706e67" />

```
missing=data[data.isnull().any(axis=1)]
missing

```

<img width="1688" height="725" alt="Screenshot 2026-05-19 013723" src="https://github.com/user-attachments/assets/97c500cb-9a68-42b4-8480-f167aeadc0e9" />

```

data2=data.dropna(axis=0)
data2

```
<img width="1664" height="737" alt="image" src="https://github.com/user-attachments/assets/43d0c03d-ebd1-422d-b696-ed0192382a88" />

```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

```
<img width="1373" height="416" alt="image" src="https://github.com/user-attachments/assets/78ebd3a6-55a1-4ad2-a48b-b524de008a5b" />

```

sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
<img width="678" height="520" alt="image" src="https://github.com/user-attachments/assets/cee1cfb7-d6cf-4277-a416-6078fd9645fd" />


```
data2
```

<img width="1539" height="575" alt="Screenshot 2026-05-19 015537" src="https://github.com/user-attachments/assets/2490a63f-20ac-478e-b910-18ea0a7092fe" />


```
new_data=pd.get_dummies(data2, drop_first=True)
new_data

```

<img width="1752" height="552" alt="Screenshot 2026-05-19 015806" src="https://github.com/user-attachments/assets/72e1c208-5bab-48aa-9933-1bdef4202f94" />

```

columns_list=list(new_data.columns)
print(columns_list)

```

<img width="1696" height="61" alt="Screenshot 2026-05-19 020223" src="https://github.com/user-attachments/assets/b3377f66-feeb-4c11-9d0b-4cb3986d37f0" />

```


features=list(set(columns_list)-set(['SalStat']))
print(features)

```
<img width="1705" height="63" alt="Screenshot 2026-05-19 020404" src="https://github.com/user-attachments/assets/bedd33b6-d220-4645-95d7-71ee489bd2e5" />

```
y=new_data['SalStat'].values
print(y)

```
<img width="276" height="48" alt="Screenshot 2026-05-19 020558" src="https://github.com/user-attachments/assets/91570803-5a23-4b02-9fc7-ab33697161c5" />

```

x=new_data[features].values
print(x)

```

<img width="524" height="175" alt="Screenshot 2026-05-19 021100" src="https://github.com/user-attachments/assets/14e68172-f12e-4cff-9e79-5e7162d51975" />

```

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)

```
<img width="427" height="106" alt="Screenshot 2026-05-19 021306" src="https://github.com/user-attachments/assets/1fb195e6-90fb-4ba9-b80b-2762ac39f706" />


```


prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)

```

<img width="257" height="80" alt="Screenshot 2026-05-19 021357" src="https://github.com/user-attachments/assets/129546ee-2dae-455a-922a-8dbd622e8edb" />

```


accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

```

<img width="312" height="45" alt="Screenshot 2026-05-19 021505" src="https://github.com/user-attachments/assets/57057ddf-8efe-4e49-8e6a-8f5db31a78b8" />

```

print("Misclassified Samples : %d" % (test_y !=prediction).sum())

```

<img width="422" height="48" alt="Screenshot 2026-05-19 021559" src="https://github.com/user-attachments/assets/09be36a2-022d-48ef-9e47-ea0db09764da" />

```

data.shape

```

<img width="212" height="71" alt="Screenshot 2026-05-19 021651" src="https://github.com/user-attachments/assets/b98fac52-481e-47d5-acd7-276876ebce87" />

```

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="1679" height="115" alt="Screenshot 2026-05-19 021817" src="https://github.com/user-attachments/assets/5185b7e1-c5d4-4010-957c-143321a84186" />

```

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

```
<img width="868" height="266" alt="Screenshot 2026-05-19 021903" src="https://github.com/user-attachments/assets/2f4ee65e-c236-4b5c-9a6a-674a8fa63fcc" />

```

tips.time.unique()

```
<img width="537" height="88" alt="Screenshot 2026-05-19 021948" src="https://github.com/user-attachments/assets/303ba477-b698-4612-8e0c-98de5cba417c" />

```

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

```
<img width="368" height="96" alt="Screenshot 2026-05-19 022032" src="https://github.com/user-attachments/assets/500f7464-0c3a-4d5d-8705-ee1751e718fe" />

```

chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")

```
<img width="483" height="109" alt="Screenshot 2026-05-19 022131" src="https://github.com/user-attachments/assets/2533780a-7076-492c-808d-54ed62ce3153" />

```

# RESULT:
  Thus, Feature selection and Feature scaling has been used on thegiven dataset.
