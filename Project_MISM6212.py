import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns


#Imputing the missing values and cleaning the dataset steps below:
file_path = 'C:/Users/User/Documents/DATAMINING6212/nyc-rolling-sales.csv'
data = pd.read_csv(file_path)
data['SALE DATE']= pd.to_datetime(data['SALE DATE'], errors='coerce')
data['sale_year'] = pd.DatetimeIndex(data['SALE DATE']).year.astype("category")
data['sale_month'] = pd.DatetimeIndex(data['SALE DATE']).month.astype("category")
pd.crosstab(data['sale_month'],data['sale_year'])
data.info()

# constructing the numerical variables:
numeric = ["RESIDENTIAL UNITS","COMMERCIAL UNITS","TOTAL UNITS", "LAND SQUARE FEET" , "GROSS SQUARE FEET","SALE PRICE" ]
for col in numeric: 
    data[col] = pd.to_numeric(data[col], errors='coerce') # coercing errors to NAs

# constructing the categorical variables:
categorical = ["BOROUGH","NEIGHBORHOOD",'BUILDING CLASS CATEGORY', 'TAX CLASS AT PRESENT', 'BUILDING CLASS AT PRESENT','ZIP CODE', 'YEAR BUILT', 'BUILDING CLASS AT TIME OF SALE', 'TAX CLASS AT TIME OF SALE']
for col in categorical:
    data[col] = data[col].astype("category")
    data.info()
    data.isna().sum()
data.replace(' ',np.nan, inplace=True) 
#replaces all empty strings (represented as ' ') in the DataFrame data with NaN (np.nan). 
#The inplace=True argument means that the changes are applied to the DataFrame in-place

data.isna().sum() /len(data) *100 #calculates the percentage of missing values for each column in the DataFrame. 
data.drop(["EASE-MENT","APARTMENT NUMBER"], axis=1, inplace=True) # remove columns of most missing values 
data=data.dropna() 
sum(data.duplicated()) # Finding duplicate values 
data.drop_duplicates(inplace=True) #dropping duplicates

#filtering the data and filtered and processed data is then saved to a new CSV file:
data[(data['SALE PRICE']<10000) | (data['SALE PRICE']>10000000)]['SALE PRICE'].count() # taking count of sales price of the given range

#only rows where the 'SALE PRICE' is between 10,000 and 10,000,000 and .copy() method is used to create a new DataFrame to avoid modifying the original data
data2= data[(data['SALE PRICE']>10000) & (data['SALE PRICE']<10000000)].copy() 

#mapped to more readable and meaningful borough names 
data2['BOROUGH']= data2['BOROUGH'].map({1:'Manhattan', 2:'Bronx', 3: 'Brooklyn', 4:'Queens',5:'Staten Island'}) 
data2.head()
data2.to_csv("NYC_New.csv", index=False)

########################################################################
# K means Clustering to create scatter plot to find Sq feet of versus sale prices:
X = data2[['SALE PRICE','GROSS SQUARE FEET']]
# Standardize the features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Determine the number of clusters (K) - You can use the elbow method to find a good value
# For this example, let's set K to 6
k = 6
# Apply K-Means clustering
kmeans = KMeans(n_clusters=k, random_state=0)
data2['cluster'] = kmeans.fit_predict(X_scaled)
# Visualize the clustered data
plt.figure(figsize=(5,5))
plt.scatter(data2['SALE PRICE'],data2['GROSS SQUARE FEET'], c=data2['cluster'], cmap='rainbow')
plt.title('K-Means Clustering')
plt.xlabel('GROSS SQUARE FEET')
plt.ylabel('SALE PRICE')
plt.show()

# box plot variation based on Borough and sale prices:
plt.figure(figsize=(12,6))
sns.boxplot(y = 'BOROUGH', x = 'SALE PRICE', data = data2 )
plt.title('Box plots for SALE PRICE on each BOROUGH')
plt.show()

#############################################################################
#K-Nearest Neighbors (k-NN) Method:
# Load your real estate data:
data2 = pd.read_csv('NYC_New.csv')
# Select independent variables and target variable
X = data2[['YEAR BUILT', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS']]  # Independent variables
y = data2['SALE PRICE']  # Target variable

# Split the data into training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize an empty list to store R-squared scores:
k_values = list(range(1, 10))
r2_scores = []

# Loop through different values of k to find the best number of neighbors:
for k in k_values:
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    r2 = knn_model.score(X_test, y_test)
    r2_scores.append(r2)
    print(f"K={k}, R-squared (R2) Score: {r2}")

# Find the best value of k with the highest R-squared score:
best_k = np.argmax(r2_scores) + 1  # Add 1 because Python uses 0-based indexing

print(f"Best K: {best_k}")
# Train the final k-NN model with the best k
final_knn_model = KNeighborsRegressor(n_neighbors=best_k)
final_knn_model.fit(X_train, y_train)
# Make predictions on the test data:
y_pred = final_knn_model.predict(X_test)
# Evaluate the model using R-squared (R2) score:
r2 = final_knn_model.score(X_test, y_test)
print(f"Final Model - R-squared (R2) Score: {r2}")

# Create a plot to visualize the relationship between k and R-squared scores:
plt.figure(figsize=(8,5))
plt.plot(k_values, r2_scores, marker='o', linestyle='-', color='b')
plt.title('R-squared (R2) vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('R-squared (R2) Score')
plt.grid(True)
plt.show()

##############################################################################

# Visualize the time series model:
data2.sort_values(by='SALE DATE', inplace=True)
# Calculate the number of rows in the DataFrame
num_rows = len(data)
# Select the first half of the rows
half_rows = num_rows // 2  # Use integer division to get the midpoint
selected_data = data.iloc[:half_rows]
plt.figure(figsize=(12, 7))
plt.plot(selected_data['sale_year'], selected_data['SALE PRICE'],label='Real Estate Prices')
plt.title('Real Estate Price Time Series Plot ')
plt.xlabel('SALE YEAR')
plt.ylabel('SALE PRICE')
plt.legend()
plt.show()

#############################################################################
