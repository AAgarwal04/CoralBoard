from utils import load_data, preprocess_data, create_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf

# Load and preprocess data
df = load_data('/home/aryan/Documents/Coral/Data.xlsx', sheet_name="Combined")
# In train_model.py
X, y = preprocess_data(df, for_training=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Normalize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test)).batch(32)

# Create and train the model
model = create_model()
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    class_weight=class_weight_dict
)

# Save the model and scaler
model.save('trained_model.h5')
import joblib
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")