from argparse import ArgumentParser # for command line arguments
from warnings import filterwarnings # for removing single class warnings from sklearn.metrics
from collections import defaultdict # for class vote counting of neighbours
from datetime import datetime # for calculating time for running the model
from typing import Iterable, List, Tuple # for static type hinting

from numpy import array, sqrt # numpy arrrays and squareroot function for distance measurement
from pandas import read_csv, DataFrame # reading and writing CSVs
from mlxtend.evaluate import bias_variance_decomp # bias variance decomposition of mean squared error estimate
from sklearn.metrics import (accuracy_score, classification_report, # goodness metrics
                             confusion_matrix)
from sklearn.model_selection import train_test_split # splitting the data into a train and a test set
from sklearn.preprocessing import MinMaxScaler # scaling data between 0 and 1

filterwarnings('ignore') # dont show single class warnings

def euclideanDistance(x: Iterable[float], y: Iterable[float]) -> float:
	'''
	:returns euclidean distance between vectors x and y upto length
	'''
	return sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)))


class KNN:
	def __init__(self, kNeighbours: int = 5):
		'''
		initialise the model with k, (default = 5 if not passed)
		'''
		self.X_train = None
		self.y_train = None
		self.k = kNeighbours

	def getKNeighbors(self, home: Iterable[float]) -> List[Tuple[Iterable[float], int]]:
		'''
		returns k nearest (based on the euclidean distance) neighbours (from the training set) to a given instance (home)
		'''
		return sorted(zip(self.X_train, self.y_train), key=lambda neighbor: euclideanDistance(home, neighbor[0]))[:self.k] # neighbour === (row(array), class(int))

	def getPrediction(self, neighbors: List[Tuple[Iterable[float], int]]) -> int:
		'''
		returns majority class of neighbors
		'''
		classVotes = defaultdict(int) # classVotes[0], [1], [2]... are all 0 initally
		for _, label in neighbors: classVotes[label] += 1 # count votes for each class label
		return max(classVotes, key=lambda x: classVotes[x]) # majority class based on vote count

	def fit(self, X_train, y_train):
		self.X_train = X_train # no real training per se, just saving the training date
		self.y_train = y_train
		return self # this is the "trained" model we return

	def predict(self, X_test):
		return array([self.getPrediction(self.getKNeighbors(home)) for home in X_test]) # predect majority class of neighbours of every row in test set

drop_columns = ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1',
                'galex_objid', 'sdss_objid', 'class', 'spectrometric_redshift', 'pred']
extinction_columns = ['extinction_g', 'extinction_i', 'extinction_r', 'extinction_u', 'extinction_z']
original_columns = ['fuv_mag', 'g', 'i', 'nuv_mag', 'r', 'u', 'z']
pairwise_columns = ['fuv-g', 'fuv-i', 'fuv-nuv', 'fuv-r', 'fuv-u', 'fuv-z', 'g-i', 'g-r', 'g-z', 'i-z', 'nuv-g', 'nuv-i', 'nuv-r', 'nuv-u', 'nuv-z', 'r-i', 'r-z', 'u-g', 'u-i', 'u-r', 'u-z']

def compute(fileName, k=5, testSize=0.3, drops=None):
	print('Starting')
	# Read datafile, first column is indices
	dataSet = read_csv(fileName, index_col=0)

	# extract spectrometric and class labels
	spectro = dataSet['spectrometric_redshift'].values
	y = dataSet['class'].values

	# drop columns not needed
	for column in drop_columns:
		try: dataSet.drop(columns=column, inplace=True)
		except: pass
	if 'P' in drops:
		for column in pairwise_columns:
			try: dataSet.drop(columns=column, inplace=True)
			except: pass
	if 'O' in drops:
		for column in original_columns:
			try: dataSet.drop(columns=column, inplace=True)
			except: pass
	if 'E' in drops:
		for column in extinction_columns:
			try: dataSet.drop(columns=column, inplace=True)
			except: pass

	# scale the data between 0 and 1
	scaler = MinMaxScaler()
	X = scaler.fit_transform(dataSet)

	# split into test and train
	X_train, X_test, y_train, y_test, spectro_train, spectro_test = train_test_split(
        X, y, spectro, test_size=testSize, random_state=0)

	# reshape single feature arrays
	spectro_train = spectro_train.reshape(-1, 1)
	spectro_test = spectro_test.reshape(-1, 1)

	print('Predicting with our KNN Model')

	knn = KNN(k)

	st = datetime.now() # start the timer
	knn.fit(X_train, y_train) # training
	y_pred = knn.predict(X_test) # testing
	tt = datetime.now() - st # total time for testing+training

	print(
		f"{accuracy_score(y_test, y_pred) * 100: .2f}% accurate | Time taken = {tt} | k = {k} | {testSize * 100}% test size | train / test = {len(X_train): 7} / {len(X_test): 7} | features = {len(X_test[0])} | {fileName.split('/')[-1]}")
	print(f"Confusion Matrix\n{confusion_matrix(y_test, y_pred)}")
	print(classification_report(y_test, y_pred))

	print('CrossValidating with the RedShift Column')

	st = datetime.now()
	spectro_pred = [0 if z <= 0.0033 else (1 if z >= 0.004 else 2) for z in spectro_test] # according to the general rule of thumb given in the base paper. class 2 is introduced for the ambigous overlapping section between 0.0033 and 0.004, since we cant definitively classify that as a star or a quasar
	tt = datetime.now() - st
	print(
		f"{accuracy_score(spectro_pred, y_pred) * 100: .2f}% accurate | Time taken = {tt} | {testSize * 100}% test size | train / test = {len(spectro_train): 7} / {len(spectro_test): 7} | features = {1} | {fileName.split('/')[-1]}")
	print(f"Confusion Matrix\n{confusion_matrix(spectro_pred, y_pred)}")
	print(classification_report(spectro_pred, y_pred))

	# print('BVD') # Bias Variance Decomposition
	# print('Avg. Expected Loss %d, Avg. Bias %d, Avg. Loss %d' % bias_variance_decomp(knn, X_train, y_train,
	                            #  X_test, y_test, num_rounds=100, random_seed=0))

	print('Saving') # Save actual, predicted and crossvalidation classes for future use
	df = DataFrame()
	df['class'] = y_test
	df['knn_pred'] = y_pred
	df['spectro_pred'] = spectro_pred
	df.to_csv(fileName.replace('.csv', '-results.tsv'), sep='\t')

	print('Done')

if __name__ == "__main__":

	parser = ArgumentParser(description='KNN Classifier')
	parser.add_argument('fileName', type=str, help='filename (required)')
	parser.add_argument('k', type=int, nargs='?', help='default=5', default=5)
	parser.add_argument('testSize', type=float, nargs='?', help='default=0.3', default=0.3)
	parser.add_argument('dropColumns', type=str, nargs='?', help='Columns to drop all but one of (OPE) Original/Pairwise/Extinction. default=None', default='')

	args = parser.parse_args()

	fileName = args.fileName
	k = args.k
	testSize = args.testSize
	drops = args.dropColumns

	compute(fileName, k, testSize, drops) # main function
