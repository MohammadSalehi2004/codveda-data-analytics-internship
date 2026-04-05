# Importing the needed libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#using this to make the background cleaner
sns.set_theme()

# The path for the csv dataset
input_file = "Data/iris.csv"

# loading the dataset
df = pd.read_csv(input_file)


# showing some information about the dataset and learning to clean it up from level 1 task 1
print("First 5 rows of the dataset:")
print(df.head(), "\n")

print("Dataset information:")
print(df.info(), "\n")

print("Missing values in each column:")
print(df.isnull().sum(), "\n")

# Removing the species column since clustering is unsupervised and since all columns are numerical
# but the species column isnt numerical
df = df.drop("species", axis=1)

# Standardizing the data so all features are on the same level and big values dont cause problems
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Kmeans elbow method to find and test the different number of clusters
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plotting the graph of the elbow method
plt.figure()

#Choosing the x and y for this plot, graph title and axis titles
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

#saving the image
plt.savefig("Level2/Plots/elbow_method.png")
plt.close()

print("Saved: elbow_method.png")


# Applying kmeans and final clustering to then visualize

# first by creating the model
kmeans = KMeans(n_clusters=3, random_state=42)

# finding patterns and goruping similar points and predicting
clusters = kmeans.fit_predict(scaled_data)

# adding cluster labels to dataset
df["Cluster"] = clusters

# Visualizing clusters by making scatter plot
plt.figure()

#Choosing the x and y for this graph, graph title and axis titles
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df["Cluster"])
plt.title("K-Means Clustering (Iris Dataset)")
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])

#saving the image
plt.savefig("Level2/Plots/iris_clusters.png")
plt.close()

print("Saved: iris_clusters.png")   