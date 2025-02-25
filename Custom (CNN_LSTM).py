from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, BatchNormalization, Flatten
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

file_path = r"C:\Users\User\Downloads\FNF_combined_1.1 - Copy.csv"

# Load the data
data = pd.read_csv(file_path)

# Replace blanks with NaN (if necessary)
data = data.replace(r'^\s*$', np.nan, regex=True)

# Apply linear interpolation to fill missing values
data = data.interpolate(method='linear', axis=0)

# Fill remaining NaNs with forward/backward fill
data = data.fillna(method='ffill').fillna(method='bfill')

# Continue with your steps for creating sequences, splitting data, etc.
X = data.drop(columns=['LABEL']).values
y = data['LABEL'].values

# Define timesteps and reshape for LSTM
timesteps = 10
num_samples = len(X) - timesteps + 1
num_features = X.shape[1]

# Create sequences for LSTM
X_lstm = np.array([X[i:i+timesteps] for i in range(num_samples)])
y_lstm = y[timesteps - 1:]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42, shuffle=True)

# # Define the Keras model
model = Sequential()
# CNN for feature extraction
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(timesteps, num_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
# model.add(Flatten())  # Flatten CNN output before passing it to LSTM

# Stacking LSTM layers with Dropout
model.add(LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))   # First LSTM layer with Dropout
model.add(LSTM(50, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))    # Second LSTM layer with Dropout
model.add(LSTM(25, return_sequences=False, dropout=0.4, recurrent_dropout=0.4))   # Third LSTM layer with Dropout

model.add(Dense(1, activation='sigmoid'))




# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))


# Reduce learning rate (e.g., from 0.001 to 0.0001)
optimizer = Adam(learning_rate=0.0001)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), class_weight=class_weights_dict, epochs=80)





# Step 1: Make predictions on the test set
y_pred_prob = model.predict(X_test)

# Step 2: Convert probabilities to binary class labels (threshold at 0.5)
y_pred = (y_pred_prob > 0.5).astype(int)

# Step 3: Generate classification report as a dictionary
report_dict = classification_report(y_test, y_pred, target_names=['Not Fall', 'Fall'], output_dict=True)

# Step 4: Convert the classification report dictionary into a pandas DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Step 5: Plot the DataFrame as a table
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust size as needed
ax.axis('tight')
ax.axis('off')
table_data = report_df.values
columns = report_df.columns
table = plt.table(cellText=table_data, colLabels=columns, rowLabels=report_df.index, loc='center', cellLoc='center', colColours=['#f5f5f5']*len(columns))


plt.savefig('E:\Paper\Fall Detection\classification_report.png', bbox_inches='tight', dpi=300)

print("Classification report saved as 'classification_report.png'.")
