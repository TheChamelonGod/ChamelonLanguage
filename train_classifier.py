import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Check data consistency
print("Original data length:", len(data_dict['data']))
print("Sample lengths:", [len(x) for x in data_dict['data'][:5]])  # Show first 5 samples

# Find the most common length
from collections import Counter
lengths = [len(x) for x in data_dict['data']]
most_common_length = Counter(lengths).most_common(1)[0][0]
print(f"Most common sequence length: {most_common_length}")

# Filter out samples with incorrect length
filtered_data = []
filtered_labels = []
for features, label in zip(data_dict['data'], data_dict['labels']):
    if len(features) == most_common_length:
        filtered_data.append(features)
        filtered_labels.append(label)
    else:
        print(f"Removed sample with length {len(features)}")

print(f"Remaining samples: {len(filtered_data)}")

# Convert to numpy arrays
data = np.array(filtered_data)
labels = np.array(filtered_labels)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{:.2f}% of samples were classified correctly!'.format(score * 100))

# Save model
with open('dist/model.p', 'wb') as f:
    pickle.dump({'model': model}, f)