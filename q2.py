from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns

#a.)
iris = load_iris()
df = pd.DataFrame(iris.data, columns= iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
#df.head()

#b.)
#Sepal Length Boxplot in pandas
sepallength = df.boxplot(column = 'sepal length (cm)', by = 'species')
sepallength
#Sepal Width
sepalwidth = df.boxplot(column = 'sepal width (cm)', by = 'species')
sepalwidth
#Petal Length
petallength = df.boxplot(column = 'petal length (cm)', by = 'species')
petallength
#Petal Width
petalwidth = df.boxplot(column = 'petal width (cm)', by = 'species')
petalwidth

#c.)
sepal = sns.scatterplot(x = 'sepal length (cm)', y = 'sepal width (cm)', data = df, hue = 'species')
petal = sns.scatterplot(x = 'petal length (cm)', y = 'petal width (cm)', data = df, hue = 'species')

#d.)
"""
For a set of rules in order to figure out classification of species type, it seems that 
    For Setosas: sepal width seems to be within 2.8-4.5, while sepal length is 4-5.8, and petal width & length are low with width being lower than .5 generally, and petal length being lower than 2 but greater than 1
    If the range of each parameter is within these boundaries, it should be classified as a setosa

We can apply the logic above for each other species, so I will not got into the specific measurements for the other species.
The method for the other species is similar, with all the classifications being within the range of measurements of the specific species, and to match a species based off of that. 

If the species in question does not fit within any specific classification type, we can use an algorithm to fit it with the ranges that are most similar to those of the other species, to provide a guess
This should be sufficient!

"""



