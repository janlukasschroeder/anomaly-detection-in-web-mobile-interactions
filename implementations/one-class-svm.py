print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
import pandas as pd

file_name = '../assets/436-Web-Mobile-Actions.csv'
# file_name = '../assets/615-Web-Mobile-Actions.csv'

df = pd.read_csv(file_name)

X = df.as_matrix(columns=df.columns[1:])

X_train = X[:-10]
X_test = X[-10:]

print X.shape[0]
print X_train.shape[0]
print X_test.shape[0]

xx, yy = np.meshgrid(np.linspace(-5, 150, 500), np.linspace(-5, 150, 500))
# Generate train data
# X = 0.3 * np.random.randn(100, 2)
# X_train = np.r_[X + 2, X - 2]
# # Generate some regular novel observations
# X = 0.3 * np.random.randn(20, 2)
# X_test = np.r_[X + 2, X - 2]
# # Generate some abnormal novel observations
# X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.0001)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
# y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
# n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection\n%s" % file_name)
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s)
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s)
# c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s)
plt.axis('tight')
plt.xlim((-5, 150))
plt.ylim((-5, 150))
plt.legend([a.collections[0], b1, b2],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper right",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "Web Actions per Day\n\n"
    "error train: %d/200 ; errors novel regular: %d/40 ; "
    % (n_error_train, n_error_test))
plt.ylabel("Mobile Action per Day")
plt.show()