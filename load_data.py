import pandas as pd
import numpy as np
from sklearn.preprocessing import  StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
import warnings
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # Enable before import
from sklearn.impute import IterativeImputer  # Now you can import it
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
# Load data
file_path = r'C:\Users\user\Pictures\Lead Scoring.xlsx'
try:
    data = pd.read_excel(file_path, engine='openpyxl')
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")

# Display first 5 rows
(data.head(5))
# Display info and statistics
data.info()
data.describe()
df = pd.read_excel(file_path)


print( "Std deviation of original page views per visit" , df['Page Views Per Visit'].dropna().std())
print( "Std deviation of original total visits" , df['TotalVisits'].dropna().std())
df.isnull().sum()


plt.hist(df["TotalVisits"], bins = 20)
plt.show()



#_____________________________________________________handing missing values__________________________________________________________


#missing values show a relation, TotalVisits and Page Views per Visit have 137 missing values. 

#furthur analysis

""" 
df["TotalVisits_Null"] = np.where(df["TotalVisits"].isnull(),1,0)
(df["TotalVisits_Null"].mean())
(df["TotalVisits_Null"])
df["Page Views Per Visit_Null"] = np.where(df["Page Views Per Visit"].isnull(),1,0)
(df["Page Views Per Visit_Null"])
(df.groupby(["TotalVisits"])["Page Views Per Visit_Null"].mean())
df.groupby(["Page Views Per Visit"])["TotalVisits_Null"].mean() """

# when page views ver visit are null, total visits are zero. when total visits are null, page views per visit are zero
#this suggests a correlation between them


# Plot heatmap, plot scatterplot 
"""
df_numeric = df.select_dtypes(include=["number"])
corr_matrix = df_numeric.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
 

sns.scatterplot(x=df["Page Views Per Visit"], y=df["TotalVisits"])
plt.title("TotalVisits vs. Page Views Per Visit")
plt.show()

"""

#heat map and scatterplot suggest weak correlation between them
#filling out missing values using iterative imputer

import gc        #removing garbage to speed up
gc.collect()


imputer = IterativeImputer()
df[['Page Views Per Visit', 'TotalVisits']] = imputer.fit_transform(df[['Page Views Per Visit', 'TotalVisits']])


print("Std deviation of new page views per visit" , df['Page Views Per Visit'].std())
print( "Std deviation of new total visits" , df['TotalVisits'].std())
#very little standard deviation observed, hence we have succesffully imputed missing vales without affecting data 


#Assuming data in Lead Source, Last Activity, Country, Industry is missing at random
#Last Activity is the most important feature but since only 103 values are missing we can use mode. 
# Rest of the categories are not going to help us detect potential leads

df["Lead Source"].mode()  #google
(df["Last Activity"].mode()) #email opened
(df["Country"].mode()) #india
(df["Industry"].mode()) #finance

df["Lead Source"].fillna("Google", inplace=True)
df["Country"].fillna("India", inplace=True)
df["Last Activity"].fillna("Email Opened", inplace=True)
df["Industry"].fillna("Finance Management", inplace=True)

(df.isnull().sum())
#__________________________________________________________feature engineering___________________________________________________________________________________________


df['Time Spent Bin'] = pd.qcut(df['Total Time Spent on Website'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])  #bining 
df = pd.get_dummies(df, columns=['Time Spent Bin'], drop_first=False) #one hot encoding




#grouping rare values together to apply one hot encoding on the above categorical values
list_5 =  df["Lead Source"].value_counts().index[:5]
df["Lead Source"] = df["Lead Source"].apply(lambda x: x if x in list_5 else "Other")
list_3 =  df["Country"].value_counts().index[:3]
df["Country"] = df["Country"].apply(lambda x: x if x in list_3 else "Other")
list_2 =  df["Lead Origin"].value_counts().index[:2]
df["Lead Origin"] = df["Lead Origin"].apply(lambda x: x if x in list_2 else "Other")
list_3 =  df["Industry"].value_counts().index[:3]
df["Industry"] = df["Industry"].apply(lambda x: x if x in list_3 else "Other")
list_5 =  df["Last Activity"].value_counts().index[:5]
df["Last Activity"] = df["Last Activity"].apply(lambda x: x if x in list_5 else "Other")

#one hot encoding
df_encoded = pd.get_dummies(df, columns=["Lead Source", "Country" , "Lead Origin" , "Industry" , "Last Activity"], 
                            prefix=["LeadSource", "Country" , "Lead Origin" , "Industry" , "Last Activity"], 
                            drop_first=False)

(df_encoded.columns)

#normalize
# 

# Define features and target
X = df_encoded.drop(columns=['Converted'])  # Features
y = df_encoded['Converted']  # Target variable

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize numerical features
scaler = StandardScaler()
X_train[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']] = scaler.fit_transform(
    X_train[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']]
)
X_test[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']] = scaler.transform(
    X_test[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']]
)

# Check the shape of training and test sets
print("Training Set Shape:", X_train.shape, y_train.shape)
print("Test Set Shape:", X_test.shape, y_test.shape)