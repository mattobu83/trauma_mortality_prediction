import pickle
import pandas as pd

# Step 1: Load the model and extract feature names
def load_model_and_features(model_path):
    # Load the trained model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Extract feature names from the model
    feature_columns = model.get_booster().feature_names

    return model, feature_columns

# Step 2: Prepare the user input for prediction
def preprocess_input(user_input, feature_columns):
    # Convert user input to a DataFrame
    input_df = pd.DataFrame([user_input], columns=feature_columns)

    # Fill missing columns with default values (e.g., 0)
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    return input_df

# Step 3: Run inference
def predict_percentage(user_input, model, feature_columns):
    # Preprocess the input
    input_df = preprocess_input(user_input, feature_columns)

    # Run prediction to get probabilities
    prediction_proba = model.predict_proba(input_df)

    # Extract the probability of the positive class (death)
    probability_of_death = prediction_proba[0][1]  # Class 1 (Death)

    # Convert to percentage
    return probability_of_death