from datetime import datetime
from multiprocessing import Pool

import pandas as pd
from numpy import average
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

csv_files = ('catalog1/cat1.csv', 'catalog2/cat2.csv', 'catalog3/cat3.csv', 'catalog4/cat4.csv')
# csv_files = ('catalog1/cat1.csv', 'catalog2/cat2.csv', 'catalog3/cat3.csv', 'catalog4/cat4.csv')
for fileName in csv_files:
	dataSet = pd.read_csv(fileName, index_col=0)

	y = dataSet['class']
	spectro = dataSet['spectrometric_redshift'].values

	for column in ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'galex_objid', 'sdss_objid', 'class', 'spectrometric_redshift', 'pred']:
		try: dataSet.drop(columns=column, inplace=True)
		except: pass

	X = StandardScaler().fit_transform(dataSet)
	k = 5
	kf = KFold(n_splits=10, random_state=29, shuffle=True)
	indexes = list(kf.split(X))

	def score(indices):
		train_index, test_index = indices
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		spectro_test = spectro[test_index]

		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(X_train, y_train)
		y_pred = knn.predict(X_test)
		spectro_pred = [0 if z <= 0.0033 else (1 if z >= 0.004 else 2) for z in spectro_test]
		return accuracy_score(y_test, y_pred), accuracy_score(spectro_pred, y_pred)

	st = datetime.now()
	scores = Pool(10).map(score, indexes)
	tt = datetime.now() - st
	print(tt, round(average([i[0] * 100 for i in scores]), 2), round(average([i[1] * 100 for i in scores]), 2), fileName)
