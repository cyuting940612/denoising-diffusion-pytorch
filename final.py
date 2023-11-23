import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Create initial data with two points
X_initial = np.array([[1, 2], [2, 3]])
y_initial = np.array([1, -1])

# Initialize and train SVM with a linear kernel
svm_classifier = SVC(kernel='linear', C=1)
svm_classifier.fit(X_initial, y_initial)


# Function to plot decision boundary and support vectors
def plot_decision_boundary_with_support_vectors(X, y, classifier, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', marker='o', s=80)

    # Highlight support vectors
    sv_indices = classifier.support_
    plt.scatter(X[sv_indices, 0], X[sv_indices, 1], c='red', marker='x', s=100, label='Support Vectors')

    # Plot decision boundary
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

    plt.title(title)
    plt.legend()
    plt.show()


# Plot initial decision boundary with two points
plot_decision_boundary_with_support_vectors(X_initial, y_initial, svm_classifier,
                                            'Initial Decision Boundary (Two Points)')

# Add a third data point
X_third = np.array([[3, 1]])
y_third = np.array([-1])

# Update and retrain the SVM with the third point
X_combined = np.vstack([X_initial, X_third])
y_combined = np.hstack([y_initial, y_third])
svm_classifier.fit(X_combined, y_combined)

# Plot decision boundary after adding the third point
plot_decision_boundary_with_support_vectors(X_combined, y_combined, svm_classifier,
                                            'Decision Boundary After Adding Third Point')