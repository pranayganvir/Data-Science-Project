# %% [markdown]
# ### Indain Used Car Price Prediction
# 
# The aim of this project to predict the price of the used cars in indian metro cities by analyzing the car's features such as company, model, variant, fuel type, quality score and many more.
# 
# #### About the Dataset
# The "Indian IT Cities Used Car Dataset 2023" is a comprehensive collection of data that offers valuable insights into the used car market across major metro cities in India. This dataset provides a wealth of information on a wide range of used car listings, encompassing details such as car models, variants, pricing, fuel types, dealer locations, warranty information, colors, kilometers driven, body styles, transmission types, ownership history, manufacture dates, model years, dealer names, CNG kit availability, and quality scores.
# 
# #### Data Dictionary
# | Column Name | Description |
# | --- | --- |
# |ID|Unique ID for each listing|
# |Company|Name of the car manufacturer|
# |Model|Name of the car model|
# |Variant|Name of the car variant|
# |Fuel Type|Fuel type of the car|
# |Color|Color of the car|
# |Killometer|Number of kilometers driven by the car|
# |Body Style|Body style of the car|
# |Transmission Type|Transmission type of the car|
# |Manufacture Date|Manufacture date of the car|
# |Model Year|Model year of the car|
# |CngKit|Whether the car has a CNG kit or not|
# |Price|Price of the car|
# |Owner Type|Number of previous owners of the car|
# |Dealer State|State in which the car is being sold|
# |Dealer Name|Name of the dealer selling the car|
# |City|City in which the car is being sold|
# |Warranty|Warranty offered by the dealer|
# |Quality Score|Quality score of the car|

# %%
# Importing the required libraries

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


# %%
# Loading the dataset
df = pd.read_csv('usedCars.csv')
df.head()

# %% [markdown]
# #### Data Preprocessing Part 1

# %%
# Shape of the dataset 
df.shape

# %%
# Columns in the dataset 
df.columns

# %%
# Dropping column ID, as it is a identifier and does not required for analysis 
df.drop('Id', axis=1, inplace=True)

# %%
df.dtypes

# %% [markdown]
# As we see Price has lakh keyword so we need to type cast it

# %%
df['Price'].head(100)

# %%
def convert_price(price):
    if 'Lakhs' in price:
        return float(price.replace('Lakhs', '').replace(',',''))* 100000
    else:
        return float(price.replace(',', ''))

# %%
df['Price'] = df['Price'].apply(convert_price)

# %%
# Checking the null values percentage wise
df.isnull().sum()/df.shape[0] * 100

# %% [markdown]
# Here in the dataset, three columns have missing values - FuelType, TransmissionType and CngKit. I will be removing the CngKit column becuase in majority of the cars don't run on CNG and the CNG cars can be easily identified from the FuelType column. So we will replace the null values with 'No' in CngKit column. In case of the TransmissionType, 67% data is missing, so we can't include this column in our analysis. In case of the FuelType, we will drop the rows with null values

# %%
df.drop('CngKit', axis = 1, inplace=True)

# %%
# Dropping TransmissionType Column 
df.drop('TransmissionType', axis=1, inplace=True)

# %%
# Removing null values from fuel ype column
df['FuelType'].dropna(inplace=True)

# %% [markdown]
# Dropping ManufacturerDate column as it the age of the car and we already have the ModelYear column

# %%
df.drop('ManufactureDate', axis=1, inplace=True)

# %%
df.drop('Variant', axis = 1, inplace=True)

# %% [markdown]
# Changing the model year column to car age column

# %%
df['ModelYear'] = 2023 - df['ModelYear']
df.rename(columns={'ModelYear':'Age'},inplace=True)

# %%
df.nunique()

# %%
df.dropna(inplace=True, axis=0)

# %%
df.isna().sum()

# %% [markdown]
# Descriptive Statistics
# 

# %%
df.describe()

# %%
df.head()

# %% [markdown]
# #### Exploratory Data Analysis
# In the exploratory data analysis, I will be looking at the distribution of data across all the columns, in order to understand the data in a better way. After that I will be looking at the relationship between the target variable and the independent variable

# %% [markdown]
# Car Company

# %%
# Number of cars by company
sns.countplot(df['Company'],order=df['Company'].value_counts().index, palette = 'Set1').set_title('Number of cars by company')
plt.show()

# %% [markdown]
# From this graph, we get know about the distribution of cars in the dataset from different companies.There are total 23 companies in the dataset, out which Maruti Suzuki, Hyundai, Honda, Mahindra and Tata are the top five companies who used cars are for sale. Therefore, we can assume that these company's car are more durable and have a good resale value.

# %% [markdown]
# Top 10 Car Models

# %%
sns.countplot(df['Model'], order=df['Model'].value_counts().iloc[:10].index, palette='Set1').set_title('Top 10 Car Models')

# %% [markdown]
# Honda City and Swift are the top two car models in the dataset, followed by Baleno, Creata and EcoSport. Therefore, we can assume that these car models are more durable and have a good resale value. Moreover, this graph also shows that Honda City and Swift are more in demand in the used car market.

# %% [markdown]
# Car Fuel Type

# %%
# Cars Count by Fuel Type
sns.countplot(x= 'FuelType', data=df, palette='Set1').set_title('Number of cars by Fuel Type')

# %% [markdown]
# Majority of cars for resale have a petrol engine which is more than 650 cars, followed by 350 cars with diesel engine. Very few of the cars have CNG engine and negligible number of cars are hybrid or on LPG. Thereofore, we can assume that petrol and diesel cars are more in demand in the used car market.

# %% [markdown]
# Top 10 Colors for Cars

# %%
#Top 10 Colors for cars
sns.countplot(x='Colour', data=df, order=df['Colour'].value_counts().iloc[:10].index, palette='Set1').set_title('Top 10 Car Colours')
plt.xticks(rotation=90)

# %% [markdown]
# Although color of car has no impact on the cars performance, but still it plays a major role in the car demand. From the graph, we can see that white color is the most preferred color for the used cars, followed by silver, grey, red and black. Therefore, we can assume that white, silver, grey, red and black color cars are more in demand in the used car market will have a good resale value.

# %% [markdown]
# Odometer Reading

# %%
#Odometer reading distrubution
sns.histplot(x = 'Kilometer', data=df, bins=20).set_title('Odometer Reading Distribution')

# %% [markdown]
# This graph shows the distribution of the odometer readings of the cars in the dataset. From the graph, we can see that most of the cars have odometer reading less than 100000 km. To be more particular majority of cars are driven for 30000 km to 50000 km. Thefore, we can assume that cars with odometer reading less than 100000 km are more in demand in the used car market will have a good resale value.

# %% [markdown]
# Body Style

# %%
#Body styole count
sns.countplot(x='BodyStyle', data=df, palette='Set1').set_title('Number of Cars by Body Style')
plt.xticks(rotation=90)

# %% [markdown]
# According to this graph, most of the cars have HatchBack, SUV and Sedan body style, which tells us about the market demand of these body styles. Therefore, we can assume that cars with HatchBack, SUV and Sedan body style are more in demand in the used car market will have a good resale value.

# %% [markdown]
# Car Age Distribution

# %%
#Car Age Distribution
sns.histplot(x='Age', data=df, bins=20).set_title('Car Age Distribution')

# %% [markdown]
# 
# 
# Age of the car plays an important role in deciding its resale value. Here, in the dataset cars that age between 5 to 7 years are more in number. Moreover majority of the cars age more than 5 years, which affect their resale value. However, there are still significant number of cars with age less than 5 years, thereofore, I assume they would have higher resale value.
# 
# In addition to that, we can see than one car has age near 20 years which could be an outlier.
# 

# %% [markdown]
# Price Distribution

# %%
# Price Distribution 
sns.histplot(x='Price', data=df, bins=30).set_title('Car Price Distribution')

# %% [markdown]
# This graph help us to know about the distribution of the car prices in the dataset. In the dataset, most of the cars have price is between 3 to 9 lakhs, with maximum cars between 3 to 6 lakhs. Therefore, we can assume that cars with price between 3 to 9 lakhs are more in demand in the used car market. Moreover there are some cars with resale price more than 20 lakhs, which could be possible for luxury cars or it could be an outlier.

# %% [markdown]
# Location based Distribution

# %%
fig, ax = plt.subplots(1,3,figsize=(20,7))

#Dealer State
sns.countplot(x = 'DealerState', data = df, ax = ax[0]).set_title('Dealer States')
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 90)

#City
sns.countplot(x = 'City', data = df, ax = ax[1]).set_title('City')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 90)

#top 10 dealers
sns.countplot(x = 'DealerName', data = df, order = df['DealerName'].value_counts().iloc[:10].index, ax = ax[2]).set_title('Top 10 Dealers')
ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation = 90)

# %% [markdown]
# These graphs shows the distribution of cars based on their dealer state, city and Dealer Name. In the dealer state graph, we see that Delhi and Maharashtra have the highest number of used cars for sale followed by Karnataka and Haryana. In the dealer city graph, we see that Delhi has the highest number of cars which is obvious from the the previous graph, however in contrast to the previous graph, Banglore has more used cars for sale than Pune, infact Pune has lower car count than Gurgaon. In the dealer name graph, we see that Car Choice Exclusif, Car&Bike Superstore Pune and Prestige Autoworld Pvt Ltd are moung the top 3 dealers with highest number of used cars for sale.

# %% [markdown]
# Car Owner Type

# %%
sns.countplot(x = 'Owner', data = df, palette='Set1').set_title('Number of cars by Owner Type')

# %% [markdown]
# The car owner type has a huge impact on its resale value. Majority of the cars that are been sold are 1at Owner cars followed by 2nd Owner cars which are significantly less in number as compared to 1st Owner. Moreover, the 3rd and 4th owner cars are very less in number. Therefore, we can assume that 1st Owner cars are more preferred in the used car market and have a good resale value.

# %% [markdown]
# Warranty

# %%
sns.countplot(x = 'Warranty', data = df, palette='Set1').set_title('Number of cars by Warranty')

# %% [markdown]
# This graphs shows the number of used cars for sale that come with a warranty from the dealership company. The warranty plays a major role and customers prefer to purchase a car with warranty, it has been shown in the dataset as well, where we can see than the number cars with warranty is almost twice the number of cars without warranty.

# %% [markdown]
# Quality Score Distribution

# %%
sns.histplot(x = 'QualityScore', data = df, bins = 10, palette='Set1').set_title('Quality Score Distribution')

# %% [markdown]
# 
# 
# Quality score is an important feature which has a huge impact on the car sales and its preference by the customers. Cars with higher quality scores tend to have a much higher resale value and are more preferred by the customers. In the dataset, most of the cars have a decent quality score between 7-8, which highlights that the cars are thoroughly checked before being sold in the used car market. However, there are some cars with quality score less than 5, which could be due to the fact that they are not in good condition or they are very old.
# 
# Till now, I have visualized the distribution of the data and got a better understanding of the data. Now, I will be looking at the relationship between the Car Price aans the independent variables.

# %% [markdown]
# Top 10 Car Companies by Price

# %%
#Top 10 car companies by price
sns.barplot(y = 'Company', x = 'Price', data = df, order = df.groupby('Company')['Price'].mean().sort_values(ascending=False).iloc[:10].index, hue = 'Company', palette= 'Set1').set_title('Top 10 car Companies by price')

# %% [markdown]
# 
# 
# This graphs highlights the top 10 car companies in the dataset with the highest resale value. The MG, Mercedes Benz and BMW are the top 3 car companies with the highest resale value, since these are luxury car companies. The list also includes Volvo. followed by KIA, Jeep and Toyota. Surprisingly Audi has much lower resale price has compared to the other luxury car companies which might be due to other features.
# 
# Moreover, my prevous hypothesis, about the car companies -Maruti Suzuki, Hyundai, Honda, Mahindra and Tata, was wrong as they are not in the top 10 list. This means that these companies cars are in greater number due to their demand because of low price
# 

# %% [markdown]
# Top 10 Car Models by Price

# %%
#Top 10 car models by price
sns.barplot(y = 'Model', x = 'Price', data = df, order = df.groupby('Model')['Price'].mean().sort_values(ascending=False).iloc[:10].index, hue = 'Model', palette= 'Set1').set_title('Top 10 car Models by price')

# %% [markdown]
# 
# 
# This graph shows the relation between the car model and it resale value and we can see that it shows similarity woth the previous graph. The car models - ML Class, Endeavour(2016_2019), CLA class are the top three models with highest resale value, followed by CLA, Fortuner and XUV700. Like the previous graph, the audi model A3 is at the 9th position with a much lower resale value as compared to the other models.
# 
# In the car model also my hypothesis was wrong as I assummed that Honda City and Swift are the top two car models in the dataset, followed by Baleno, Creata and EcoSport. Therefore, we came to know that these car in higher number due to their high demnad because of low price.
# 

# %% [markdown]
# Car Fuel Type and Price

# %%
fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.boxplot(x = 'FuelType', y = 'Price', data = df, ax = ax[0], hue = 'FuelType').set_title('Price by Fuel Type')
sns.violinplot(x = 'FuelType', y = 'Price', data = df, ax = ax[1], hue = 'FuelType').set_title('Price by Fuel Type')

# %% [markdown]
# The above plots visualizes the relationship between the car fuel type and its resale value. In the boxplot we can see than cars with diesel fuel type have higher resale value than petrol and CNG and LPG. In the violin plot, we can see that the distribution of the price for diesel cars is more concentrated between 10 to 20 lakh as compared to Petrol. From this it is cleared that, customers prefer petrol and diesel car than other fuel type and the diesel cars are more in demand in the used car market.

# %% [markdown]
# Top 10 Car Colors by Price

# %%
#Top 10 car colors by price
sns.barplot(y = 'Colour', x = 'Price', data = df, order = df.groupby('Colour')['Price'].mean().sort_values(ascending=False).iloc[:10].index).set_title('Top 10 car Colors by price')

# %% [markdown]
# 
# 
# The cars with colors like Burgundy, Riviera Red and Dark Blue have higher resale value as compared to other colors. This shows that color of the car does matter and plays a major role in the resale value of the car.
# 
# Moreover, we also came to know that exotic colors have more price but they are not in demand in the used car market.
# 

# %% [markdown]
# Odometer Reading and Price

# %%
sns.scatterplot(x = 'Kilometer', y = 'Price', data = df).set_title('Odometer Reading and Price')

# %% [markdown]
# In the scatter plot we can see than the data is concentrated near the origin, which means that most of the cars have odometer reading less than 100000 km. In addition to that the cars with less odometer reading shows higher resale value and as the odometer reading increases the resale value decreases. Therefore, my hypothesis was correct that cars with odometer reading less than 100000 km are more in demand in the used car market will have a good resale value.

# %% [markdown]
# Body Style and Price

# %%
sns.barplot(x = 'BodyStyle', y = 'Price', data = df, hue = 'BodyStyle').set_title('Price by Body Style')
plt.xticks(rotation = 90)

# %% [markdown]
# MPV, SUV and Sedan are the top 3 car body styles with the highest resale value. Therefore, we can assume that these body styles are more preferred in the used car market and have a good resale value. This also shows that my assumption was correct however, the Hatchback body style cars despite being in majority have lower resale value.

# %% [markdown]
# Car Age and Price

# %%
sns.barplot(x = 'Age', y = 'Price', data = df).set_title('Car age and Price')



# %% [markdown]
# As we discussed earlier, age is a key determinant for a car's resale value and this graph clearly visulaizes the relation of the age with car price. The cars with age less than a year has then highest price and as the age increases the prices decreases gradually. Therefore, my hypothesis was correct that cars with age less than 5 years have higher resale value.

# %% [markdown]
# Location based Price Distribution

# %%
fig, ax = plt.subplots(1,3,figsize=(20,7))

#Dealer State
sns.violinplot(x = 'DealerState', y = 'Price', data = df, ax = ax[0], hue = 'DealerState').set_title('Dealer States')
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 90)

#City
sns.violinplot(x = 'City',y = 'Price', data = df, ax = ax[1], hue = 'City').set_title('City')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 90)

#top 10 dealers
sns.violinplot(x = 'DealerName',y = 'Price', data = df, order = df['DealerName'].value_counts().iloc[:10].index, ax = ax[2], hue = 'DealerName').set_title('Top 10 Dealers')
ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation = 90)

# %% [markdown]
# In the above graph we can see the price distribution based on the state, city and the dealer name. In the state graph, we can see that the cars in Rajastan have the highest price followed by Delhi. Moreover, there are some outliers in the graph which os visible from the violinplot where there is strong peak incase of Haryana. In the city graph, we can see that the cars in Jaipur have the highest price followed by Mumbai and Delhi. Moreover, there are some outliers in the graph which os visible from the violinplot where there is strong peak incase of Gurgaon. In the dealer name graph, we can see the top 10 dealers along with their price distribution. Here, Car Estate has the highest price followed by Star Auto India and Car Choice. Moreover, there are some outliers in the graph which os visible from the violinplot where there is strong peak incase of Noida Car Ghar.

# %% [markdown]
# Car Owner Type and Price

# %%
sns.violinplot(x = 'Owner', y = 'Price', data = df, hue = 'Owner').set_title('Price by Owner Type')

# %% [markdown]
# 
# 
# The graph shows the price distribution with respect to the car owner type. The cars with 1st owner have the highest price which is obvious as they are new cars. However, the 3rd Owner type cars depite being less in number have higher price than 2nd Owner type cars, which is not obvious. Therefore, we can assume that 3rd Owner type cars having higher price could some luxury or vintage cars.
# 

# %% [markdown]
# 
# Warranty and Price

# %%
sns.violinplot(x = 'Warranty', y = 'Price', data = df, hue = 'Warranty').set_title('Price by Warranty')

# %% [markdown]
# Here, we can see some change in the violinplot of the cars with and without warranty. The cars with warranty tends to have slightly higher price than the cars without warranty. Therefore, we can assume that cars with warranty are more preferred in the used car market and have a good resale value.

# %% [markdown]
# Quality Score and Price

# %%
sns.scatterplot(x = 'QualityScore', y = 'Price', data = df).set_title('Quality Score and Price')

# %% [markdown]
# We can see a very high concentration near the quality score 7 and above having much higher price than the cars with quality score less than 7. Therefore, we can assume that cars with quality score 7 and above are more preferred in the used car market and have a good resale value.

# %% [markdown]
# Data Preprocessing Part 2

# %% [markdown]
# Dropping column car model beacause, it has too many unique values and it will increase the dimensionality of the dataset.

# %%
df.drop('Model', axis = 1, inplace = True)

# %% [markdown]
# Label Encoding
# 

# %%
#columns for label encoding
cols = df.select_dtypes(include=['object']).columns

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

#Label encoding for object type columns 
for i in cols:
    le.fit(df[i])
    df[i]= le.transform(df[i])
    print(i, df[i].unique())

# %% [markdown]
# Outlier Removal

# %%
# Using quantile method to remove outliers

# Columns for outlier removal
cols = df.select_dtypes(include=['float64', 'int64']).columns
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

# Removing outliers
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) | (df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# %% [markdown]
# #Correlation Matrix Heatmap

# %%
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True)

# %%
# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Price', axis=1), df['Price'], test_size=0.2, random_state=42)

# %% [markdown]
# # Model Building
# I will be using the following regression models:
# - Decision Tree Regressor
# - Random Forest Regressor
# - Ridge Regressor

# %% [markdown]
# Decision Tree Regressor

# %%
from sklearn.tree import DecisionTreeRegressor

# Decision Tree Regressor
dtr = DecisionTreeRegressor()

# %% [markdown]
# Hyperparameter Tuning
# 

# %%
from sklearn.model_selection import GridSearchCV

#parameters for hyperparameter tuning
para = {
    'max_depth': [2,4,6,8,10],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf' : [2,4,6,8,10],
    'random_state':[0, 42]
}


# Grid Search Object
grid = GridSearchCV(estimator=dtr, param_grid=para, cv=5, n_jobs=-1, verbose=1)

# Fitting the model
grid.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid.best_params_)

# %%
#Decision Tree Regressor wiht best parameters
dtr = DecisionTreeRegressor(max_depth=8, min_samples_leaf=2, min_samples_split=8, random_state=42)

# Fitting the model
dtr.fit(X_train, y_train)

#Training score
print("Training Score:", dtr.score(X_train, y_train))

# %%
# Prediction 
dtr_pred = dtr.predict(X_test)

# %% [markdown]
# Model Metrics

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %%
#Decision Tree Regressor
print('Decision Tree Regressor')
print('Mean Squared Error : ', mean_squared_error(y_test, dtr_pred))
print('Mean Absolute Error : ', mean_absolute_error(y_test, dtr_pred))
print('R2 Score : ', r2_score(y_test, dtr_pred))

# %% [markdown]
# Random Forest Regressor

# %%
from sklearn.ensemble import RandomForestRegressor

# Random Forest Regressor
rfr = RandomForestRegressor()

# %% [markdown]
# Hyperparameter Tuning

# %%
#parameters for hyperparameter tuning
para ={
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [2, 4, 6, 8, 10],
    'random_state': [0, 42]
}

# Grid Search Object
grid = GridSearchCV(estimator=rfr, param_grid=para, cv=5, n_jobs=-1, verbose=1)

# Fitting the model
grid.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid.best_params_)

# %%
#Random Forest Regressor with best parameters
rfr = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_leaf=2, min_samples_split=2, random_state=42)

#Fitting the model
rfr.fit(X_train, y_train)

#Training score
print(rfr.score(X_train, y_train))

# %%
# Prediction
rfr_pred = rfr.predict(X_test)

# %%
#Random Forest Regressor
print('Random Forest Regressor')
print('Mean Squared Error : ', mean_squared_error(y_test, rfr_pred))
print('Mean Absolute Error : ', mean_absolute_error(y_test, rfr_pred))
print('R2 Score : ', r2_score(y_test, rfr_pred))

# %% [markdown]
# Model Evaluation
# 

# %% [markdown]
# Distribution Plot

# %%
fig,ax = plt.subplots(1,2,figsize=(10,5))

#decision tree regressor
sns.distplot(x = y_test, ax = ax[0], color = 'r', hist = False, label = 'Actual').set_title('Decision Tree Regressor')
sns.distplot(x = dtr_pred, ax = ax[0], color = 'b', hist = False, label = 'Predicted')

#random forest regressor
sns.distplot(x = y_test, ax = ax[1], color = 'r', hist = False, label = 'Actual').set_title('Random Forest Regressor')
sns.distplot(x = rfr_pred, ax = ax[1], color = 'b', hist = False, label = 'Predicted')

# %% [markdown]
#  Feature Importance

# %%
fig, ax = plt.subplots(1,2,figsize=(15, 5))
fig.subplots_adjust(wspace=0.5)

#Decision Tree Regressor
feature_df = pd.DataFrame({'Features':X_train.columns, 'Importance':dtr.feature_importances_})
feature_df.sort_values(by='Importance', ascending=False, inplace=True)
sns.barplot(x = 'Importance', y = 'Features', data = feature_df, ax = ax[0]).set_title('Decision Tree Regressor')

#Random Forest Regressor
feature_df = pd.DataFrame({'Features':X_train.columns, 'Importance':rfr.feature_importances_})
feature_df.sort_values(by='Importance', ascending=False, inplace=True)
sns.barplot(x = 'Importance', y = 'Features', data = feature_df, ax = ax[1]).set_title('Random Forest Regressor')

# %% [markdown]
# ### Conclusion
# 
# 
# 
# From the exploratory data analysis, I have revealed two major facts about the used car market: which are demand and price. The demand of low price used car is pretty high as compared to the to expensive ones, which highlights the customers attraction towards budget cars. But upon studying the graph I also came to know about some interesting facts about the used car market. Begining with the car companies, companies like- MG, Mercedes Benz, BMW, Volvo and KIA have the highest price but Maruti Suzuki, Hyundai, Honda, Mahindra and Tata car are in higher demand. This highlights that customer prefer to buy new luxury cars instead of used ones.
# 
# Majority of the cars run either on petro or diesel, with diesel cars having slightly higher price. I als came to know that car is major player in the market. Cars like white, grey, silver and black are in higher demand but exotic colors like burgundy, riviera red, dark blue, black majic have higher price. Coming to the car's odometer reading, most of the cars have reading less than 10,000 km, and cars with lower odometer reading have the higher price.
# 
# Cars with bodystyle like HatchBack, SUV and Sedan are most preferred by the customers whereas the bodystyle like MPV, SUV and Sedan are the top most ecpensive ones. Age of the car also play a major role in its resale value. As the car age increases, it resale value decreases. Therefore, cars than age less than 5 years have higher price and prefered more. Car price aslo changes by location. Delhi, Maharashtra and Rajstan are the top three states with the highes price and Car Estate, Star Auto India and Car Choice are the top three dealers with the highest price.
# 
# Customers usually prefer the car with 1st owner type resulting in hugher demand as well as higher price. Cars that comes with a warranty provudes an assurance to the customer, resulting in a little bit higher price. The last feature i.e. Quality score also dictates the car price, where cars with higher quality score have higher price.
# 
# Coming to the machine learning models, I have used Descision tree regressor and random forest regressor to predict the car price. The random forest regressor model performed better than the decision tree regressor model. Moreover, from the feature importance graph, we can see that the car age, bodystyle and comapny are the key features that affect the car price.
# 

# %% [markdown]
# 


