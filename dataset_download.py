"""
Downloading dataset:

First step: Obtain API Credentials

- Log in to your Kaggle account.

- Navigate to your account settings (click your avatar in the top right, then select "Account").

- Scroll down to the "API" section and click "Create New API Token." This will download a kaggle.json file containing your API key and username

- Create a folder called .kaggle in the same place where this .py file is. Put the .json file in the folder so that the path is ~/.kaggle/kaggle.json

Instructions from : https://www.kaggle.com/docs/api
"""

import os
import kaggle

os.chdir('..') # Should be in the directory right outside the repository
os.makedirs('BRATS20', exist_ok=True)


# Import and authenticate
kaggle.api.authenticate()

# Download and unzip dataset
kaggle.api.dataset_download_files(
    'awsaf49/brats20-dataset-training-validation', 
    path='BRATS20', 
    unzip=True
)
