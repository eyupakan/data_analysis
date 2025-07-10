# import libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# load dataset (digits)
digits = load_digits()
X = digits.data  # features
y = digits.target  # target veriable

# visualization
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for i, ax in enumerate(axes):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"digits.target[i]")
    ax.axis("off")
plt.show()


# data train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# define model
gb_clf = GradientBoostingClassifier(
    n_estimators=150,  # ağaç sayısı
    learning_rate=0.05,  # küçük öğrenme oranı
    max_depth=4,  # ağaç derinliği
    subsample=0.8,  # her ağacı verinin %80'i ile eğit
    min_samples_split=5,  # dallanma yada bölünme için gereken min  sample sayısı
    min_samples_leaf=3,  # yaprakta en az 3 örnek olmalı
    max_features="sqrt",  # özelliklerin karekökü kadar kullan
    validation_fraction=0.1,  # erken durdurma için %10  validation
    n_iter_no_change=5,  # 5 iterasyon boyunca iyileşme yoksa training'i yarıda kes
    random_state=42,
)

# model training
gb_clf.fit(X_train, y_train)


# testing
y_pred = gb_clf.predict(X_test)

# evaluation
print(f"GB Accuracy: {accuracy_score(y_test,y_pred)}")
print(f"GB Classification Report: \n{classification_report(y_test,y_pred)}")
