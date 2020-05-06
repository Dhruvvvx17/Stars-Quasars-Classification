from collections import defaultdict
from multiprocessing import Pool
from sys import argv

from numpy import array, sqrt, average
from pandas import read_csv, DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

def euclideanDistance(x, y): return sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)))

class KNN:
	def getKNeighbors(self, home):
		return sorted(zip(self.X_train, self.y_train), key=lambda neighbor: euclideanDistance(home, neighbor[0]))[:5]

	def getPrediction(self, neighbors):
		classVotes = defaultdict(int)
		for _, label in neighbors: classVotes[label] += 1
		return max(classVotes, key=lambda x: classVotes[x])

	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_test):
		return array([self.getPrediction(self.getKNeighbors(home)) for home in X_test])

fileName = argv[1]

dataSet = read_csv(fileName, index_col=0)

spectro = dataSet['spectrometric_redshift'].values
y = dataSet['class'].values

for column in ('Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'galex_objid', 'sdss_objid', 'class', 'spectrometric_redshift', 'pred'):
	try: dataSet.drop(columns=column, inplace=True)
	except: pass

X = MinMaxScaler().fit_transform(dataSet)

kf = KFold(n_splits=10, random_state=29, shuffle=True)
indexes = list(kf.split(X))

def score(indices):
	train_index, test_index = indices
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	spectro_test = spectro[test_index]

	knn = KNN()

	knn.fit(X_train, y_train)
	y_pred = knn.predict(X_test)
	spectro_pred = [0 if z <= 0.0033 else (1 if z >= 0.004 else 2) for z in spectro_test]
	return accuracy_score(y_test, y_pred), accuracy_score(spectro_pred, y_pred)

scores = Pool(10).map(score, indexes)
print(fileName, round(average([i[0] * 100 for i in scores]), 2), round(average([i[1] * 100 for i in scores]), 2))
