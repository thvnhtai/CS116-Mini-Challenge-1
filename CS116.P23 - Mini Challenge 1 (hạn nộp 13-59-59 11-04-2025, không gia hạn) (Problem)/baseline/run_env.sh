#!/bin/bash

echo "Setting up environment for ML competition..."

# Install required packages for improved models
pip install scikit-learn==1.4.0 xgboost==2.0.3 lightgbm==4.1.0 numpy==1.24.4 pandas==2.1.4 joblib==1.3.2
pip install scikit-optimize==0.9.0

echo "Environment setup completed!"
