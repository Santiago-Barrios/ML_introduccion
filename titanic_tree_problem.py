import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

test_df = pd.read_csv('./titanic-test.csv')
train_df = pd.read_csv('./titanic-train.csv')

print(train_df.info())
train_df.Sex.value_counts().plot(kind = 'bar', color = ['b', 'r'])
plt.title('Distribucion de sobrevivientes')
plt.show()

# pasar una variable textual a una númerica
print(train_df.head(2))
label_encoder = preprocessing.LabelEncoder()
encoder_sex = label_encoder.fit_transform(train_df['Sex'])
# train_df['Sex']= encoder_sex
print(train_df.head(2))


# Llenar espacios vacíos en las columnas Age y Embarked
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna('S')

# Predictores eliminar columnas
train_predictors = train_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)

categorical_cols = [cname for cname in train_predictors.columns if
                        train_predictors[cname].nunique() < 10 and
                        train_predictors[cname].dtype == 'object'
                    ]

numerical_cols = [cname for cname in train_predictors.columns if
                    train_predictors[cname].dtype in ['int64', 'float64']
                ]

my_cols = categorical_cols + numerical_cols
train_predictors = train_predictors[my_cols]

dummy_encoded_train_predictors = pd.get_dummies(train_predictors)
print(dummy_encoded_train_predictors)
print(train_df['Pclass'].value_counts())

y_target = train_df['Survived'].values
x_features_one = dummy_encoded_train_predictors.values

X_train, x_validation, Y_train, y_validation = train_test_split(x_features_one, y_target, test_size=.25, random_state=1 )

# árbol de desición con los datos entrenados
tree_one = tree.DecisionTreeClassifier()
tree_one = tree_one.fit(X_train, Y_train)

# comprobación del accuracy
tree_one_accuracy = round(tree_one.score(x_validation,y_validation), 4)
print(tree_one_accuracy)

# Mostrar el arbol de decisión
from io import StringIO
from IPython.display import Image, display
import pydotplus

out = StringIO()
tree.export_graphviz(tree_one, out_file = out)

graph = pydotplus.graph_from_dot_data(out.getvalue())
graph.write_png('titanic.png')