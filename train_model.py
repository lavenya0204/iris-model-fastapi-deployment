import pandas as pd
from sklearn.datasets import load_iris    #The Iris dataset is classic and comes pre-loaded with scikit-learn
from sklearn.model_selection import train_test_split    #To split data into traing and testing data
from sklearn.linear_model import LogisticRegression    #A classifier
import joblib #For saving and loading the model efficiently
import os 

print("Starting model training process...")

# 1. Load the Iris Dataset
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
Y = iris.target # Target (species: 0-setosa, 1-versicolor, 2-virginica)
target_names = iris.target_names # To map numerical values to names of target classes

print("Iris dataset loaded successfully.")
print(f"Features shape: {X.shape}")
print(f"Target shape: {Y.shape}")
print(f"Target names: {target_names}")


# 2. Split the Data into Training and Testing Sets
# test_size=0.2 means 20% of the data will be used for testing and the rest 80% for training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("\nData split into training and testing sets:")
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")


# 3. Logistic Regression is a good choice for multi-class classification 
model = LogisticRegression(max_iter=200) # Increased max_iter for robustness

print("\nTraining Logistic Regression model...")
model.fit(X_train, Y_train)
print("Model training complete.")

# 4. Evaluating the Model to test how well the model performs on test data
accuracy = model.score(X_test, Y_test)
print(f"\nModel Accuracy on test set: {accuracy:.2f}")

# 5. Save the Trained Model using Joblib
# It's a good practice to save the model so we don't have to retrain it every time
# the FastAPI app starts

# Create a directory for models if it doesn't exist
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created directory: {models_dir}")

model_filename = os.path.join(models_dir, 'iris_logistic_regression_model.joblib')
joblib.dump(model, model_filename)

print(f"\nModel saved successfully to: {model_filename}")

# Also save the target names for later use in FastAPI
target_names_filename = os.path.join(models_dir, 'iris_target_names.joblib')
joblib.dump(target_names.tolist(), target_names_filename) # .tolist() to save as a standard list
print(f"Target names saved successfully to: {target_names_filename}")

print("\nModel creation script finished.")