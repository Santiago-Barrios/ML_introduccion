import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

diabetes = pd.read_csv('./datasets/diabetes.csv')
print(diabetes.head(5))
print(diabetes.shape)

features_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
x = diabetes[features_cols]
y = diabetes.Outcome

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_pred = logreg.predict(X_test)

print(y_pred)


cnf_matrix = metrics.confusion_matrix(Y_test, y_pred)
print(cnf_matrix)

class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap="Blues_r", fmt="g")
ax.xaxis.set_label_position("bottom")
plt.tight_layout()
plt.title("Matriz de confusión", y=1)
plt.ylabel("Etiqueta actual")
plt.xlabel("Etiqueta de predicción")
plt.show()

print("exactitud", metrics.accuracy_score(Y_test, y_pred))