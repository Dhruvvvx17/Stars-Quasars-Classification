from datetime import datetime

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

csv_files = ('catalog1/cat1.csv', 'catalog2/cat2.csv', 'catalog3/cat3.csv', 'catalog4/cat4.csv')
for fileName in csv_files:
	dataSet = pd.read_csv(fileName, index_col=0)

	d_class = dataSet['class']

	for column in ['Unnamed: 0.1.1', 'Unnamed: 0.1', 'Unnamed: 0', 'galex_objid', 'sdss_objid', 'class', 'spectrometric_redshift', 'pred']:
		try: dataSet.drop(columns=column, inplace=True)
		except: pass

	dataSet[dataSet.columns] = MinMaxScaler().fit_transform(dataSet[dataSet.columns])

	X_train, X_test, y_train, y_test = train_test_split(dataSet, d_class, test_size=0.3, random_state=0)

	k = 5
	knn = KNeighborsClassifier(n_neighbors=k)
	st = datetime.now()
	knn.fit(X_train, y_train)
	y_pred = knn.predict(X_test)
	print(datetime.now() - st, round(metrics.accuracy_score(y_test, y_pred) * 100, 2), fileName)
