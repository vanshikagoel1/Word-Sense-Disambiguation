import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset with BERT embeddings
df = pd.read_csv('semcor_dataset_with_embeddings.csv')

# Encode senses into numerical labels
label_encoder = LabelEncoder()
df['sense_label'] = label_encoder.fit_transform(df['sense'])

# Split dataset into training and testing sets
X = np.stack(df['embedding'].values)
y = df['sense_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save trained classifier
import joblib
joblib.dump(clf, 'wsd_classifier.pkl')
