#importing core pkgs
import streamlit as st 

#importing datasets pkgs
from sklearn import datasets

#importing classifiers pkgs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#importing training & testing pkgs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#importing utilities pkgs
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#functions implemintations

def get_dataset(dataset_name):
	if dataset_name == "Iris dataset":
		data = datasets.load_iris()

	elif dataset_name == "Breast cancer dataset":
		data = datasets.load_breast_cancer()

	else:
		data = datasets.load_wine()

	x = data.data
	y = data.target

	return x, y

def add_parameter_ui(clf_name):
	params = dict()
	if clf_name == "KNN":
		K = st.sidebar.slider("K",1,15)
		params["K"] = K
	elif clf_name == "SVM":
		C = st.sidebar.slider("C",0.01,10.0)
		params["C"] = C
	else:
		max_depth = st.sidebar.slider("max_depth",2,15)
		n_estimators = st.sidebar.slider("n_estimators",1,100)
		params["max_depth"] = max_depth
		params["n_estimators"] = n_estimators
	return params

def get_classifier(clf_name, params):
	if clf_name == "KNN":
		clf = KNeighborsClassifier(n_neighbors = params["K"])

	elif clf_name == "SVM":
		clf = SVC(C = params["C"])

	else: 
		clf = RandomForestClassifier(n_estimators = params["n_estimators"],
									 max_depth = params["max_depth"],
									 random_state = 4)
	return clf

st.title("Learn Easy ML Algorithms Application")

st.write("""
# Explor different classifier
Which one is the best?
	""")
menu = st.sidebar.selectbox("Menu",("Home", "About"))
if menu == "About":
	st.write("""
 created based on tutorial from python engineer youtube channel with some minor changes i applied
""")
	st.write("""alittle demo about me :\n
I am Mostafa moaaz ,love python and ML applications\n
love streamlit and its powerful and effortlessly way of creating such awesome looking webpages\n
feel free to take a look at my personal accounts\n
linked in account : https://www.linkedin.com/in/mostafa-moaaz-fcih/\n
github account : https://github.com/mostafamoaaz\n
	""")
else:
	dataset_name = st.sidebar.selectbox("Select Dataset",("Iris dataset","Breast cancer dataset","Wine dataset"))

	classifier_name = st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random Forest"))

	#calling the dataset requested by user and show its shape and its classes numbers
	x, y = get_dataset(dataset_name)
	st.write("shape of dataset", x.shape)
	st.write("number of classes", len(np.unique(y)))

	#calling the parameters for the chosen classifier ,the parameters are slider bar
	params = add_parameter_ui(classifier_name)

	# using the parameters and the classifier ;the classifier is called
	clf = get_classifier(classifier_name,params)

	#classification

	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)

	clf.fit(X_train, y_train)

	y_pred = clf.predict(X_test)

	#measuring the accuracy metrics
	acc = accuracy_score(y_test, y_pred)
	st.write(f"classifier = {classifier_name}")
	st.write(f"accuracy = {acc}")

	#using PCA to reduce matrices space degree into 2d dimentional  "princibal component analysis"

	pca = PCA(2)

	x_projected = pca.fit_transform(x)

	x1 = x_projected[:,0]
	x2 = x_projected[:,1]

	# ploting the results using pyplot lib

	fig = plt.figure()
	plt.scatter(x1, x2, c=y, alpha = 0.8, cmap = "viridis")
	plt.xlabel("Principal Component 1")
	plt.ylabel("Principal Component 2")

	plt.colorbar()

	#plt.show()
	st.pyplot(fig)