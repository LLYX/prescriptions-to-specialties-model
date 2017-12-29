import sklearn
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold='nan')

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm, trange

def cm(labels, preds, labels_array):
	cm = confusion_matrix(labels, preds)

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	plt.imshow(cm, cmap='hot', interpolation='nearest')
	cax = ax.matshow(cm)
	fig.colorbar(cax)
	plt.ylabel('Correct Class')
	plt.xlabel('Predicted Class')
	plt.xticks(np.arange(1, len(labels_array) + 1, 1.0))
	plt.yticks(np.arange(1, len(labels_array) + 1, 1.0))
	plt.grid()
	plt.show()

def tuned_hyperparameters_forest():
	forest = ExtraTreesClassifier()

	params = {'n_estimators': [120, 300, 500, 800, 1200], 'max_depth': [5, 8, 15, 25, 30, None], 'min_samples_split': [1, 2, 5, 10, 15, 100], 'min_samples_leaf': [1, 2, 5, 10], 'max_features': ['log2', 'sqrt', None]}

	return GridSearchCV(forest, params)

def tree_feature_selection(train_X, train_y, test_X):
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(train_X, train_y)

    model = SelectFromModel(forest, prefit=True)

    train_X_new = model.transform(train_X)
    test_X_new = model.transform(test_X)

    return train_X_new, test_X_new

def tsvd(train_X, test_X):
	tsvd = TruncatedSVD(n_components=200)

	train_X_new = tsvd.fit_transform(train_X)
	test_X_new = tsvd.transform(test_X)

	return train_X_new, test_X_new

def get_data(file):
	features = []
	labels = []

	# Getting raw features from the data file

	with open(file, 'r') as infile:
		for line in tqdm(infile):
			if line[0] != 'n':
				line = line.rstrip('\r\n').split('\t')

				if line[16] == '':
					line[16] = 0.0
				else:
					line[16] = float(line[16])

				features.append(line[3:5] + line[7:9] + [float(x) if x != '' else 0.0 for x in line[9:15]] + [line[15]] + [line[16]] + [line[17]] + [float(x) if x != '' else 0.0 for x in line[18:]])
				labels.append(line[5])

	features, labels = np.array(features), np.array(labels)

	categorical_features_indices = [0, 1, 2, 3, 10, 12]

	oh_enc = OneHotEncoder(categorical_features=categorical_features_indices)

	l_enc = LabelEncoder()

	# Convert all categorical data into numerical data

	features[:, 0] = l_enc.fit_transform(features[:, 0])

	features[:, 1] = l_enc.fit_transform(features[:, 1])

	features[:, 2] = l_enc.fit_transform(features[:, 2])

	features[:, 3] = l_enc.fit_transform(features[:, 3])

	features[:, 10] = l_enc.fit_transform(features[:, 10])

	features[:, 12] = l_enc.fit_transform(features[:, 12])

	# One hot encode the categorical features

	features = oh_enc.fit_transform(features)

	labels = l_enc.fit_transform(labels)

	return train_test_split(features, labels, test_size=0.3) + [l_enc]

def main():
	train_X, test_X, train_y, test_y, l_enc = get_data('test.txt')

	# train_X, test_X = tsvd(train_X, test_X)

	print('Selecting features...')

	train_X, test_X = tree_feature_selection(train_X, train_y, test_X)

	# model = tuned_hyperparameters_forest()

	# model.fit(train_X, train_y)

	# preds = model.predict(test_X)

	# acc = (preds == test_y).mean()

	# print(acc)

	# print(model.best_params_)

	print('Training model...')

	model = ExtraTreesClassifier(n_estimators=120, bootstrap=True, oob_score=True, max_features='sqrt')

	model.fit(train_X, train_y)

	print('Predicting...')

	preds = model.predict(test_X)

	acc = (preds == test_y).mean()

	print(acc)

	print(model.oob_score_)

	enc_labels_array = np.unique(test_y)

	labels_array = np.unique(l_enc.inverse_transform(test_y))

	print(zip(enc_labels_array, labels_array))

	cm(l_enc.inverse_transform(test_y), l_enc.inverse_transform(preds), labels_array)

if __name__ == '__main__':
	main()