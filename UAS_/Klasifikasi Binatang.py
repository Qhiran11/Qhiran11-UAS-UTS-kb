import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pandas as pd


data_file = r"C:\Users\YOGA\Documents\1. Semester 3\Kecerdasan Buatan\UAS_\data_fisik_labels.csv"
data = pd.read_csv(data_file)


data_fisik = data["Data Fisik"].tolist()
labels = data["Label"].tolist()


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data_fisik)


X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42)


k_values = range(1, 11)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    accuracies.append(accuracy_score(y_val, y_pred))


plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.title('K-NN Performance')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


best_k = k_values[np.argmax(accuracies)]
print(f"Best k: {best_k}")
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)


y_pred = knn.predict(X_val)
print("\nClassification Report:")
print(classification_report(y_val, y_pred))


def klasifikasi_binatang(ciri_fisik):
    input_data = vectorizer.transform([ciri_fisik])
    prediksi = knn.predict(input_data)
    return prediksi[0]


input_ciri = input("Masukkan ciri fisik binatang (misal: berbulu, terbang, paruh): ")
hasil_klasifikasi = klasifikasi_binatang(input_ciri)
print(f"Jenis binatang yang dimaksud adalah: {hasil_klasifikasi}")
