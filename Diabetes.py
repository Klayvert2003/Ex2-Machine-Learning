import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score

#Carregar o conjunto de dados Iris
df = pd.read_csv('diabetes.csv')
df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI',
              'DiabetesPedigreeFunction', 'Age', 'Outcome']

X = df.drop('DiabetesPedigreeFunction', axis=1)
y = df['Outcome']

# Dividir o conjunto de dados em conjuntos de treinamento (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar uma instância do classificador de árvore de decisão
clf = DecisionTreeClassifier(random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
print("Acurácia Média:", scores.mean())
# Treinar o classificador usando os dados de treinamento
clf.fit(X_train, y_train)
DecisionTreeClassifier(random_state=42)

# Prever os rótulos das amostras de teste
y_pred = clf.predict(X_test)

# Calcular métricas de avaliação
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
confusion_mat = confusion_matrix(y_test, y_pred)

print("\n", "Acurácia:", accuracy, "\n", "Precisão:", precision, "\n", "Revocação(Recall):", recall, "\n", "F1-Score:", f1, "\n", "Matriz de Confusão:\n", confusion_mat, "\n")