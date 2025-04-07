import argparse
import sys
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
import xgboost as xgb
import lightgbm as lgb
from joblib import dump, load
import time
from sklearn.metrics import accuracy_score

# Defines a function for training the model.
def run_train(public_dir, model_dir):
    # Ensures the model directory exists, creates it if it doesn't.
    os.makedirs(model_dir, exist_ok=True)
    
    print("Starting the training process...")
    start_time = time.time()

    # Constructs the path to the training data file.
    train_file = os.path.join(public_dir, 'train_data', 'train.npz')

    # Loads the training data from the .npz file.
    train_data = np.load(train_file)

    # Extracts the features and labels from the training data
    X_full = train_data['X_train']
    y_full = train_data['y_train']
    
    # Split into training and validation sets (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    
    print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
    print(f"Classes in training data: {np.unique(y_train, return_counts=True)}")
    
    # Create preprocessing pipeline
    preprocessor = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95, random_state=42))  # Keep 95% of variance
    ])
    
    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    print(f"After preprocessing - Training data shape: {X_train_processed.shape}")
    
    # Create a set of models
    models = {
        'logistic': LogisticRegression(max_iter=1000, C=1.0, solver='liblinear', random_state=42),
        'rf': RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42),
        'xgb': xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42),
        'lgb': lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42),
    }
    
    # Create an ensemble model with the best performing models
    ensemble = VotingClassifier(
        estimators=[
            ('rf', models['rf']),
            ('xgb', models['xgb']),
            ('gb', models['gb'])
        ],
        voting='soft'
    )
    
    # Evaluate models with cross-validation
    print("Evaluating individual models...")
    best_model = None
    best_score = 0
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        val_score = accuracy_score(y_val, model.predict(X_val_processed))
        print(f"{name} validation accuracy: {val_score:.4f}")
        
        if val_score > best_score:
            best_score = val_score
            best_model = model
    
    # Train and evaluate ensemble model
    print("Training ensemble model...")
    ensemble.fit(X_train_processed, y_train)
    ensemble_score = accuracy_score(y_val, ensemble.predict(X_val_processed))
    print(f"Ensemble validation accuracy: {ensemble_score:.4f}")
    
    # Select the best performing model (either individual or ensemble)
    if ensemble_score > best_score:
        best_model = ensemble
        best_score = ensemble_score
    
    print(f"Best model: {best_model.__class__.__name__} with accuracy: {best_score:.4f}")
    
    # Train final model on entire dataset
    print("Training final model on entire dataset...")
    X_full_processed = preprocessor.fit_transform(X_full)
    best_model.fit(X_full_processed, y_full)
    
    # Save the preprocessor and model
    print("Saving model and preprocessor...")
    dump(preprocessor, os.path.join(model_dir, 'preprocessor.joblib'))
    dump(best_model, os.path.join(model_dir, 'model.joblib'))
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds")


# Defines a function for making predictions.
def run_predict(model_dir, test_input_dir, output_path):
    print("Starting prediction process...")
    start_time = time.time()
    
    # Load preprocessor and model
    preprocessor = load(os.path.join(model_dir, 'preprocessor.joblib'))
    model = load(os.path.join(model_dir, 'model.joblib'))
    
    # Constructs the path to the test data file.
    test_file = os.path.join(test_input_dir, 'test.npz')
    
    # Loads the test data from the .npz file.
    test_data = np.load(test_file)
    
    # Extracts the features from the test data.
    X_test = test_data['X_test']
    
    print(f"Test data shape: {X_test.shape}")
    
    # Preprocess the test data
    X_test_processed = preprocessor.transform(X_test)
    print(f"After preprocessing - Test data shape: {X_test_processed.shape}")
    
    # Make predictions
    y_pred = model.predict(X_test_processed)
    
    # Save predictions
    pd.DataFrame({'y': y_pred}).to_json(output_path, orient='records', lines=True)
    
    print(f"Prediction completed in {time.time() - start_time:.2f} seconds")


# Defines the main function that parses commands and arguments.
def main():
    # Initializes a parser for command-line arguments.
    parser = argparse.ArgumentParser()

    # Creates subparsers for different commands.
    subparsers = parser.add_subparsers(dest='command')

    # Adds a subparser for the 'train' command.
    parser_train = subparsers.add_parser('train')

    # Adds an argument for the directory containing public data.
    parser_train.add_argument('--public_dir', type=str)

    # Adds an argument for the directory to save the model.
    parser_train.add_argument('--model_dir', type=str)

    # Adds a subparser for the 'predict' command.
    parser_predict = subparsers.add_parser('predict')

    # Adds an argument for the directory containing the model.
    parser_predict.add_argument('--model_dir', type=str)

    # Adds an argument for the directory containing test data.
    parser_predict.add_argument('--test_input_dir', type=str)

    # Adds an argument for the path to save prediction results.
    parser_predict.add_argument('--output_path', type=str)

    # Parses the command-line arguments.
    args = parser.parse_args()

    if args.command == 'train':
        # Checks if the 'train' command was given.
        # Calls the function to train the model.
        run_train(args.public_dir, args.model_dir)
    elif args.command == 'predict':
        # Checks if the 'predict' command was given.
        # Calls the function to make predictions.
        run_predict(args.model_dir, args.test_input_dir, args.output_path) 
    else:
        # If no valid command was given, prints the help message.
        # Displays help message for the CLI.
        parser.print_help()

        # Exits the script with a status code indicating an error.
        sys.exit(1)


# Checks if the script is being run as the main program.
if __name__ == "__main__":
    # Calls the main function if the script is executed directly.
    main()
