import data_loader
from sklearn.mixture import GaussianMixture
import numpy as np
import util
import argparse

parser = argparse.ArgumentParser(description="Train Gaussian Mixture Model to Classify Acoustic Scene")
parser.add_argument('--data_dir', type=str, default="./data", help="Train Dataset Directory")
parser.add_argument('--cov_type', type=str, default="diag", help="Choose Covariance Type From ['spherical', 'diag', 'tied', 'full']")
parser.add_argument('--max_iter', type=int, default=100, help="Max Number of Iteration")
opt = parser.parse_args()

dataset = data_loader.load_data(opt.data_dir)

X_train, Y_train, X_test, Y_test = data_loader.train_and_test(dataset)
n_classes = 4
# n_classes = np.unique(Y_train)..?

'''
estimators = {cov_type: GaussianMixture(n_components=n_classes, covariance_type=cov_type,
        max_iter=100, random_state=0) for cov_type in ['spherical', 'diag', 'tied', 'full']}

for index, (name, estimator) in enumerate(estimators.items()):
    estimator.means_init = np.array([X_train[Y_train==i].mean(axis=0) for i in range(n_classes)])
    print(name, "cov")
    
    estimator.fit(X_train)
    
    y_train_pred = estimator.predict(X_train)
    train_acc = np.mean(y_train_pred.ravel() == Y_train.ravel()) * 100
    
    y_test_pred = estimator.predict(X_test)
    test_acc = np.mean(y_test_pred.ravel() == Y_test.ravel()) * 100

    print("train accuracy :", train_acc)
    print("test accuracy :", test_acc)
'''

estimator = GaussianMixture(n_components=n_classes, covariance_type=opt.cov_type, max_iter=opt.max_iter, random_state=0, verbose=1)

estimator.means_init = np.array([X_train[Y_train==i].mean(axis=0) for i in range(n_classes)])

estimator.fit(X_train)
y_train_pred = estimator.predict(X_train)
train_acc = np.mean(y_train_pred.ravel() == Y_train.ravel()) * 100

y_test_pred = estimator.predict(X_test)
test_acc = np.mean(y_test_pred.ravel() == Y_test.ravel()) * 100

print("train accuracy: ", train_acc)
print("test accuracy: ", test_acc)
util.save_model(estimator)
