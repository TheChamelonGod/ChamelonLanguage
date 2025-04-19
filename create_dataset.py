import os
import pickle
import cv2
import numpy as np
import mediapipe as mp

# Suppress TensorFlow and MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    max_num_hands=1
)

DATA_DIR = './data'

# Create mapping from directory names to numeric labels (0-25)
label_mapping = {str(i): i for i in range(26)}  # Maps '0'-'25' to numbers

data = []
labels = []
expected_features = 42  # 21 landmarks * 2 coordinates


def numeric_sort_key(s):
    """Simple numeric sorting for filenames"""
    try:
        return int(os.path.splitext(s)[0])  # Try to convert filename (without extension) to number
    except ValueError:
        return float('inf')  # Non-numeric filenames will go to the end


# Verify all expected directories exist
missing_dirs = [str(i) for i in range(26) if not os.path.exists(os.path.join(DATA_DIR, str(i)))]
if missing_dirs:
    print(f"Warning: Missing directories for labels: {', '.join(missing_dirs)}")

# Get and sort directories numerically
dir_list = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
dir_list.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))

for dir_name in dir_list:
    dir_path = os.path.join(DATA_DIR, dir_name)

    if dir_name not in label_mapping:
        print(f"Skipping unexpected directory: {dir_name}")
        continue

    numeric_label = label_mapping[dir_name]
    print(f"\nProcessing label {numeric_label} (directory: {dir_name})")
    processed_images = 0

    # Process images in directory
    img_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    img_files.sort(key=numeric_sort_key)

    for img_file in img_files:
        img_path = os.path.join(dir_path, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"  Could not read image: {img_file}")
            continue

        # Process image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            print(f"  No hand detected in: {img_file}")
            continue

        data_aux = []
        x_ = []
        y_ = []

        # Process first hand only
        hand_landmarks = results.multi_hand_landmarks[0]

        # Collect landmarks
        for landmark in hand_landmarks.landmark:
            x_.append(landmark.x)
            y_.append(landmark.y)

        # Normalize coordinates
        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))

        # Validate feature vector length
        if len(data_aux) != expected_features:
            print(f"  Unexpected feature length {len(data_aux)} in {img_file}")
            continue

        data.append(data_aux)
        labels.append(numeric_label)
        processed_images += 1
        print(f"  Processed: {img_file}")

    print(f"Completed label {numeric_label}: {processed_images} images")

# Save the dataset
dataset = {
    'data': np.array(data, dtype=np.float32),
    'labels': np.array(labels),
    'label_names': [chr(65 + i) for i in range(1)]  # A-Z labels
}

with open('sign_language_dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)

print("\nDataset creation completed!")
print(f"Total samples: {len(data)}")
print("Label counts:")
for label in range(26):
    count = list(labels).count(label)
    if count > 6:
        print(f"  {label} ({chr(65 + label)}): {count} samples")