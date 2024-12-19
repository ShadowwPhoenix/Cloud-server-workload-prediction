# CPU Usage Prediction with LSTM

## Overview
This project leverages a Long Short-Term Memory (LSTM) neural network to predict CPU usage based on various input features from a dataset of virtual machine (VM) statistics. The model processes time-series data and provides predictions for CPU usage in the future based on past usage trends and other features.

## Files
- `vmCloud_data.csv`: The raw dataset used for training the model.
- `saved_lstm_model.keras`: The trained LSTM model saved for future use.
- `cpu_usage_prediction.py`: The script that loads the dataset, preprocesses the data, trains the LSTM model, and evaluates its performance.

## Data Preprocessing
The data is preprocessed as follows:
1. **Drop unnecessary columns**: `timestamp`, `vm_id`.
2. **Handle missing values**: Numeric columns are filled with their mean values.
3. **One-hot encoding**: Applied to categorical columns (`task_type`, `task_priority`, `task_status`).
4. **Feature engineering**: Create lag features (`cpu_usage_lag1`, `cpu_usage_lag2`, ...) and rolling statistics (mean, standard deviation, min, max).
5. **Data Scaling**: The data is scaled using `StandardScaler` to standardize the features.

## Model Architecture
The model is a sequential LSTM neural network with the following layers:
1. **LSTM Layer**: 150 units with return sequences.
2. **Dropout Layer**: Dropout rate of 0.4 to prevent overfitting.
3. **LSTM Layer**: 100 units without returning sequences.
4. **Dropout Layer**: Dropout rate of 0.4.
5. **Dense Layer**: Single neuron to output the predicted `cpu_usage`.

## Model Training
The model is trained with the following configuration:
- **Optimizer**: Adam with a learning rate of 0.0005.
- **Loss Function**: Mean Squared Error (MSE).
- **Epochs**: 100 epochs.
- **Batch Size**: 64.

## Model Evaluation
The model performance is evaluated using the following metrics:
- **Mean Absolute Error (MAE)**: 0.45
- **Mean Squared Error (MSE)**: 0.32
- **Root Mean Squared Error (RMSE)**: 0.56
- **R-squared (R2)**: 0.9996

## Predictions vs Actual Values (First 10 Predictions)

| Predicted | Actual |
|-----------|--------|
| 18.07     | 17.64  |
| 65.71     | 64.98  |
| 28.13     | 27.89  |
| 83.53     | 83.17  |
| 68.62     | 68.42  |
| 3.08      | 2.52   |
| 96.17     | 95.58  |
| 93.27     | 93.32  |
| 92.26     | 92.01  |
| 70.94     | 70.28  |

## Visualization
An interactive plot is generated using `plotly` to compare actual vs predicted CPU usage for the first 100 test samples.

## Saving and Loading the Model
The trained LSTM model is saved using Keras' `save_model` function and can be loaded back for future predictions.

## How to Use
1. **Load the trained model**:
    ```python
    from tensorflow.keras.models import load_model
    model = load_model('saved_lstm_model.keras')
    ```

2. **Prepare the input data** (following the same preprocessing steps as in the training phase).

3. **Make predictions**:
    ```python
    y_pred = model.predict(X_test_scaled)
    ```

4. **Evaluate the model's performance** using metrics like MAE, MSE, RMSE, and R2.

## License
This project is licensed under the MIT License.
