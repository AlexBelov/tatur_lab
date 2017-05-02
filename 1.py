import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, svm, datasets
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, precision_score, recall_score

n_neighbors = 2
n_train = 5

# import some data to play with
iris = datasets.load_iris()
train_indices = np.random.choice(range(100), n_train)
X = iris.data[:-50, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
X_train = map(lambda i: X[i], train_indices)
y = iris.target[:-50]
y_train = np.take(y, train_indices)

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# options: uniform, distance
weights = 'uniform'

# we create an instance of Neighbours Classifier and fit the data.
knn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
knn.fit(X_train, y_train)

C = 1.0  # SVM regularization parameter
lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)

for i, clf in enumerate((knn, lin_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_classes = clf.predict(X)
    # pdb.set_trace()
    fpr, tpr, thresholds = roc_curve(y, Z_classes)

    if i == 0:
        print("KNN")
    else:
        print("SVM")

    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)

    f_score = f1_score(y, Z_classes, average='binary')
    print("F score: %f" % f_score)

    p_score = precision_score(y, Z_classes)
    print("Precision score: %f" % p_score)

    r_score = recall_score(y, Z_classes)
    print("Recall score: %f" % r_score)

    plt.figure(1)
    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.figure(2*(i+1) + 1)
    # Plot clusterization
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
                % (n_neighbors, weights))

    # Plot confusion matrix
    cm = confusion_matrix(y, Z_classes)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.show()
