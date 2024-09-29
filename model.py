import pickle  # Import pickle for saving the model

import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

# Load the preprocessed Titanic dataset
df = pd.read_csv("/content/titanic_preprocessed.csv")

# Step 1: Define Features and Target
# Assuming 'Survived' is the target column
X = df.drop(columns=["Survived"])  # Features
y = df["Survived"]  # Target (Survival label)

# Step 2: Split the Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize TPOT Classifier (AutoML)
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)

# Step 4: Train the AutoML Model
tpot.fit(X_train, y_train)

# Step 5: Evaluate the Model on the Test Set
accuracy = tpot.score(X_test, y_test)
print(f"TPOT AutoML Model Accuracy: {accuracy}")

# Step 6: Generate Predictions and Evaluation Metrics
y_pred = tpot.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Export the Best Model as Python Code
tpot.export("best_titanic_pipeline.py")
print("Best model exported to 'best_titanic_pipeline.py'")

# NEW Step: Save the TPOT model as a pickle file
with open("best_titanic_model.pkl", "wb") as f:
    pickle.dump(tpot.fitted_pipeline_, f)  # Save the fitted pipeline
print("Best model saved to 'best_titanic_model.pkl'")
