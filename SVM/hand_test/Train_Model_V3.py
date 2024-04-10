import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Define different len_hand values
len_hands = [1, 2]   

for len_hand in len_hands:
    if len_hand == 1:
        dataset_path = "dataset_onehand_V3"
        label_file = "labels1_V3.txt"
    elif len_hand == 2:
        dataset_path = "dataset_twohand_V3"
        label_file = "labels2_V3.txt"

    # Fetch all .npy files from the dataset directory
    files = [f for f in os.listdir(dataset_path) if f.endswith('.npy')]
    # print(files)

    X = []
    y = []

    labels = []

    # Load data and labels
    for idx, file in enumerate(files):
        data = np.load(os.path.join(dataset_path, file))
        X.extend(data)  # Note: using extend instead of append
        label = file.split('.')[0]
        labels.append(label)
        y.extend([idx] * data.shape[0])

    X = np.array(X)
    y = np.array(y)

    # Save labels to labels.txt
    with open(label_file, 'w') as f:
        for label in labels:
            f.write(label + "\n")

    # Preprocess data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Check if model exists, if not, train a new model
    model_filename = f"svm_model_V3_len_hand_{len_hand}.pkl"
    scaler_filename = f"scaler_V3_len_hand_{len_hand}.pkl"

    if os.path.exists(model_filename) and os.path.exists(scaler_filename):
        # Load existing model and scaler
        clf = joblib.load(model_filename)
        saved_scaler = joblib.load(scaler_filename)
        # Update scaler with new data
        X = saved_scaler.transform(X)
        # Incremental training
        clf.fit(X, y)
    else:
        # Train new model
        clf = SVC(kernel='linear', probability=True)
        clf.fit(X, y)

        # Save the model and the scaler
        joblib.dump(clf, model_filename)
        joblib.dump(scaler, scaler_filename)
        print(f"Model saved to {model_filename}")
        print(f"Scaler saved to {scaler_filename}")
        print(f"Labels saved to {label_file}")
        