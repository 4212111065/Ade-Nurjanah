# Install library yang diperlukan
!pip install scikit-image
!pip install scikit-learn

# Import library yang diperlukan
from sklearn import datasets
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np

# Load dataset MNIST
digits = datasets.load_digits()

# Mendapatkan HOG features dan labels
hog_features = []
labels = []

for img, label in zip(digits.images, digits.target):
    fd = hog(img, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), block_norm='L2-Hys')
    hog_features.append(fd)
    labels.append(label)

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)

# Membuat model SVM
clf = SVC(kernel='linear')

# Melatih model
clf.fit(X_train, y_train)

# Memprediksi data uji
y_pred = clf.predict(X_test)

# Evaluasi performa
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')

# Menampilkan hasil
print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)
print("Precision:", precision)
