###################### Imports #################
# Basic scientific computing and visualization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
# Machine learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
# Data splitting and model evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
import copy

# Read in data
data = pd.read_csv("C:/Users/RDITLRCP/Desktop/Misc/ERDC/Self-Projects/Concrete-Neural-Network/concrete.csv")

# Create a copied dataframe to use for visualization.
copied_data = data.copy(deep=True)

# Convert the data to numpy arrays of features and targets
data = data.to_numpy()

# Training parameters
epochs = 1000
learning_rate = 0.001

# Data 
x = data[:,0:8]
y = data[:,8]

# Split into training and testing sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

class ConcreteDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(data=x, dtype=torch.float32) # Input features
        self.y = torch.tensor(data=y, dtype=torch.float32).unsqueeze(1) # Output features
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        sample = {"x": self.x[index], "y": self.y[index]}
        return sample
    
train_dataset = ConcreteDataset(x=x_train, y=y_train)
test_dataset = ConcreteDataset(x=x_test, y=y_test)

train_data_loader = DataLoader(dataset=train_dataset)
test_data_loader = DataLoader(dataset=test_dataset)

# Class Early_Stopping
class Early_Stopping:
    def __init__(self, patience=10, min_delta=0.0001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""
        
    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False


# Class Regression_Model
class Regression_Model(nn.Module):
    def __init__(self, input_columns):
        super(Regression_Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=input_columns, out_features=10),
            nn.LeakyReLU(),
            nn.Linear(in_features=10, out_features=50),
            nn.LeakyReLU(),
            nn.Linear(in_features=50, out_features=100),
            nn.LeakyReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.LeakyReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.LeakyReLU(),
            nn.Linear(100, 1)
        )
    
    def forward(self, x):
        return self.model(x)
    
model = Regression_Model(input_columns=x_train.shape[1])
early_stopping = Early_Stopping()

def train_test_model(model, train_data_loader, test_data_loader, epochs, learning_rate):
    # Loss Function
    criterion = nn.MSELoss()
    # Optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # Create lists to store the loss history
    history = {'train_loss': [], 'test_loss': []}

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        model.train()
        running_training_loss = 0.0
        for batch_idx, sample_batched in enumerate(train_data_loader):
            inputs = sample_batched['x']
            labels_train = sample_batched['y']
            # Backpropagation and optimization
            optimizer.zero_grad()
            y_pred_train = model(inputs)
            loss_train = criterion(y_pred_train, labels_train)
            loss_train.backward()
            optimizer.step()
            running_training_loss += loss_train.item()
        avg_train_loss = running_training_loss / len(train_data_loader)
        
        history['train_loss'].append(avg_train_loss)

        # Model testing
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(test_data_loader):
                inputs_test = sample_batched['x']
                labels_test = sample_batched['y']
                y_pred_test = model(inputs_test)
                loss_test = criterion(y_pred_test, labels_test)
                running_test_loss += loss_test.item()

        avg_test_loss = running_test_loss / len(test_data_loader)

        history['test_loss'].append(avg_test_loss)

        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f} - Status: {early_stopping.status}")

        if early_stopping(model, avg_test_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break # Exit the training loop
            
    print("Training finished.")

    return history # <--- ADD THIS
    
training_history = train_test_model(model=model, train_data_loader=train_data_loader, test_data_loader=test_data_loader, epochs=epochs, learning_rate=learning_rate)
# Evaluate the model for prediction.
model.eval()
with torch.no_grad():
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test_predict = model(x_test)

# --- Step 1: Set up the multi-run experiment ---
NUM_RUNS = 3
all_runs_metrics = {'MSE': [], 'RMSE': [], 'MSLE': [], 'RMSLE': [], 'MAE': []}
all_runs_history = [] # To store the learning curves from each run
all_runs_residuals = {'train': [], 'test': []} # To store residuals for Q-Q plots

for i in range(NUM_RUNS):
    print(f"\n{'='*20} STARTING RUN {i+1}/{NUM_RUNS} {'='*20}")

    model = Regression_Model(input_columns=x_train.shape[1])
    early_stopping = Early_Stopping()

    training_history = train_test_model(
        model=model,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        epochs=epochs,
        learning_rate=learning_rate
    )
    all_runs_history.append(training_history)

    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_predict = model(x_test_tensor)

    y_actual = y_test
    y_predicted = y_test_predict.detach().numpy().flatten()

    # --- Calculate and store metrics ---
    mse = mean_squared_error(y_actual, y_predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_predicted) # NEW: Calculate MAE
    
    y_predicted[y_predicted < 0] = 0
    msle = mean_squared_log_error(y_actual, y_predicted)
    rmsle = np.sqrt(msle)

    all_runs_metrics['MSE'].append(mse)
    all_runs_metrics['RMSE'].append(rmse)
    all_runs_metrics['MSLE'].append(msle)
    all_runs_metrics['RMSLE'].append(rmsle)
    all_runs_metrics['MAE'].append(mae) # NEW: Store MAE

    # --- Calculate and store residuals ---
    with torch.no_grad():
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_predict = model(x_train_tensor).detach().numpy().flatten()
    train_residuals = y_train - y_train_predict
    test_residuals = y_actual - y_predicted
    all_runs_residuals['train'].append(train_residuals)
    all_runs_residuals['test'].append(test_residuals)

# =================================================================================
# === SELF-CONTAINED PLOTTING BLOCK 1: ACCURACY BY RANGE PLOT (ROBUST VERSION) ====
# =================================================================================
# --- Step 1: Data Preparation ---
df_results = pd.DataFrame({'Actual': y_actual, 'Predicted': y_predicted})
relative_tolerance = 0.05
df_results['relative_error'] = np.abs(df_results['Predicted'] - df_results['Actual']) / df_results['Actual']
df_results['within_tolerance'] = df_results['relative_error'] <= relative_tolerance

# --- Step 2: Define bins and create the categorical column ---
bins = [0, 20, 40, 60, df_results['Actual'].max() + 1]
labels = ['0-20 MPa', '20-40 MPa', '40-60 MPa', '60+ MPa']
# Create as a categorical type so we can use 'observed=False'
df_results['strength_bin'] = pd.cut(df_results['Actual'], bins=bins, labels=labels, right=False)

# --- Step 3: Group data using 'observed=False' to include empty bins ---
# THIS IS THE CRITICAL FIX that prevents the loop from stopping.
# It tells pandas to create a group for every label, even if no data falls into it.
accuracy_by_bin = df_results.groupby('strength_bin', observed=False)['within_tolerance'].mean().fillna(0) * 100
counts_by_bin = df_results.groupby('strength_bin', observed=False).size()

# --- Step 4: Create, Plot, and Save the Figure ---
fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
accuracy_by_bin.plot(kind='bar', ax=ax_acc, color='steelblue', edgecolor='k', rot=0)
ax_acc.set_title(f'Model Accuracy (within {relative_tolerance:.0%}) by Concrete Strength Range', fontsize=16)
ax_acc.set_ylabel('Percent of Predictions within Tolerance', fontsize=12)
ax_acc.set_xlabel('Actual Concrete Strength Range (MPa)', fontsize=12)
ax_acc.set_ylim(0, 105)
ax_acc.grid(axis='y', linestyle='--', alpha=0.7)

# --- Step 5: Annotate Bars ---
for i, p in enumerate(ax_acc.patches):
    count = counts_by_bin.iloc[i]
    if count > 0:
        label = f'{p.get_height():.1f}%\n(N={count})'
        ax_acc.annotate(label, (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontsize=10)

# --- Step 6: Save and explicitly CLOSE THIS FIGURE ---
plt.tight_layout()
plt.savefig(f'accuracy_by_range_run_{i+1}.png', dpi=300, bbox_inches='tight')
print(f"Saved 'accuracy_by_range_run_{i+1}.png'")
plt.close(fig_acc)

# (Your Q-Q plot block should follow immediately after this, unchanged)

# =================================================================================
# === SELF-CONTAINED PLOTTING BLOCK 2: Q-Q PLOTS ==================================
# =================================================================================
# This block already followed good practice, so it is mostly unchanged.

# --- Create a figure with 1 row and 2 columns for the side-by-side plots ---
fig_qq, axes_qq = plt.subplots(1, 2, figsize=(14, 6))
# fig_qq.suptitle(f'Q-Q Plots for Run {i+1}', fontsize=18)

# --- Determine a common scale for both plots for direct comparison ---
combined_residuals = np.concatenate([train_residuals, test_residuals])
min_val, max_val = combined_residuals.min(), combined_residuals.max()
buffer = (max_val - min_val) * 0.05
plot_range = [min_val - buffer, max_val + buffer]

# --- Plot 1: Training Residuals (Left side) ---
ax1 = axes_qq[0]
stats.probplot(train_residuals, dist="norm", plot=ax1)
ax1.set_title('Training Set Errors')
ax1.set_xlabel('Theoretical Quantiles')
ax1.set_ylabel('Sample Quantiles (Errors)') # Residuals
ax1.grid(True)
ax1.set_xlim(plot_range); ax1.set_ylim(plot_range)
ax1.set_aspect('equal', adjustable='box')

# --- Plot 2: Testing Residuals (Right side) ---
ax2 = axes_qq[1]
stats.probplot(test_residuals, dist="norm", plot=ax2)
ax2.set_title('Testing Set Errors')
ax2.set_xlabel('Theoretical Quantiles')
ax2.set_ylabel('Sample Quantiles (Errors)')
ax2.grid(True)
ax2.set_xlim(plot_range); ax2.set_ylim(plot_range)
ax2.set_aspect('equal', adjustable='box')

# --- Save the figure with a unique name and close it ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'qq_plot_run_{i+1}.png', dpi=300)
print(f"Saved 'qq_plot_run_{i+1}.png'")
plt.close(fig_qq) # Close the specific figure we just made

# ==============================================================================
# --- END OF THE FOR LOOP ---
# ==============================================================================

print(f"\n{'='*20} ALL RUNS COMPLETED {'='*20}")
print("\n--- Final Metrics Summary Across All Runs ---")
for metric, values in all_runs_metrics.items():
    formatted_values = [f"{v:.4f}" for v in values]
    print(f"{metric}: {formatted_values}")
print("------------------------------------------\n")


# --- PLOT 1: Learning Curves for All Runs ---
plt.figure(figsize=(12, 7))
colors = plt.cm.viridis(np.linspace(0, 1, NUM_RUNS))
for i, history in enumerate(all_runs_history):
    plt.plot(history['test_loss'], linestyle='-', color=colors[i], label=f'Run {i+1} Test Loss')
    plt.plot(history['train_loss'], linestyle='--', color=colors[i], alpha=0.6, label=f'Run {i+1} Train Loss')
plt.title('Training & Test Loss Across Multiple Runs', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.ylim(bottom=0)
plt.tight_layout()
# plt.savefig('learning_curves_comparison.png', dpi=300, bbox_inches='tight')
print("Saved 'learning_curves_comparison.png'")


# --- PLOT 2 & 3: Comparison Bar Charts for Final Metrics ---
run_labels = [f'Run {i+1}' for i in range(NUM_RUNS)]
# Plot for MSE
plt.figure(figsize=(10, 6))
bars_mse = plt.bar(run_labels, all_runs_metrics['MSE'], color=colors)
plt.title(f'Comparison of Final MSE Scores Across {NUM_RUNS} Runs', fontsize=16)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
for bar in bars_mse:
    plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height(), f'{bar.get_height():.2f}', va='bottom', ha='center')
# plt.savefig('mse_comparison_barchart.png', dpi=300, bbox_inches='tight')
print("Saved 'mse_comparison_barchart.png'")
# Plot for RMSE
plt.figure(figsize=(10, 6))
bars_rmse = plt.bar(run_labels, all_runs_metrics['RMSE'], color=colors)
plt.title(f'Comparison of Final RMSE Scores Across {NUM_RUNS} Runs', fontsize=16)
plt.ylabel('Root Mean Squared Error (RMSE) in MPa', fontsize=12)
for bar in bars_rmse:
    plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height(), f'{bar.get_height():.2f}', va='bottom', ha='center')
# plt.savefig('rmse_comparison_barchart.png', dpi=300, bbox_inches='tight')
print("Saved 'rmse_comparison_barchart.png'")

# --- NEW PLOT FOR MAE ---
plt.figure(figsize=(10, 6))
bars_mae = plt.bar(run_labels, all_runs_metrics['MAE'], color=colors)
plt.title(f'Comparison of Final MAE Scores Across {NUM_RUNS} Runs', fontsize=16)
plt.ylabel('Mean Absolute Error (MAE) in MPa', fontsize=12)
for bar in bars_mae:
    plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height(), f'{bar.get_height():.2f}', va='bottom', ha='center')
# plt.savefig('mae_comparison_barchart.png', dpi=300, bbox_inches='tight')
print("Saved 'mae_comparison_barchart.png'")
# --- END OF NEW PLOT ---

# Show all plot windows on screen
# plt.show()


# ==============================================================================
#                 --- CODE FOR PLOTTING AND EVALUATION ---
#           Place this entire block after your prediction line (line 165)
# ==============================================================================

# --- Step 1: Convert Tensors to NumPy Arrays for Plotting ---
"""# The 'y_test' variable is already a NumPy array from your train_test_split.
# We need to convert the model's prediction tensor to a NumPy array.
# .detach() removes it from the computation graph.
# .numpy() converts it to a NumPy array.
# .flatten() changes its shape from (n, 1) to (n,) for easier plotting.
"""
y_actual = y_test
y_predicted = y_test_predict.detach().numpy().flatten()


# --- Step 2: Calculate the Error Metrics ---
# Note: MSLE and RMSLE can only be used if values are non-negative.
# This is safe for concrete strength.

mse = mean_squared_error(y_actual, y_predicted)
rmse = np.sqrt(mse)
msle = mean_squared_log_error(y_actual, y_predicted)
rmsle = np.sqrt(msle)

# Store metrics in a dictionary for easy plotting later
error_metrics = {
    'MSE': mse,
    'RMSE': rmse
}

# Print the final metric results to the console
print("\n--- Model Evaluation Metrics ---")
for name, value in error_metrics.items():
    print(f'{name}: {value:.4f}')
print("--------------------------------\n")


# --- Step 3: Generate and Save All Plots ---

# =========================================================================
# PLOT 1: The Standard - Scatter Plot of Actual vs. Predicted (The Best Choice)
# =========================================================================
# WHY IT WORKS: This is the best way to visualize regression results. The closer
# the points are to the 45-degree red line, the better the model's predictions.
# It clearly shows the overall performance and any systematic biases.

plt.figure(figsize=(8, 8))
plt.scatter(y_actual, y_predicted, alpha=0.6, edgecolors='k', label='Predictions')

# Enforcing the "Same Scale" Rule by finding the min/max of all data
min_val = min(y_actual.min(), y_predicted.min())
max_val = max(y_actual.max(), y_predicted.max())
plot_range = [min_val - 5, max_val + 5]

# Add the 45-degree perfect prediction line
plt.plot(plot_range, plot_range, 'r--', lw=2, label='Perfect Prediction')

plt.title('Actual vs. Predicted Concrete Strength', fontsize=16)
plt.xlabel('Actual Compressive Strength (MPa)', fontsize=12)
plt.ylabel('Predicted Compressive Strength (MPa)', fontsize=12)
plt.xlim(plot_range)
plt.ylim(plot_range)
plt.grid(True)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box') # Makes the plot perfectly square
# plt.savefig('actual_vs_predicted_scatter.png', dpi=300, bbox_inches='tight')
print("Saved 'actual_vs_predicted_scatter.png'")


# =========================================================================
# PLOT 2: Line Plot of Predictions vs. Actuals (Generally a Bad Choice)
# =========================================================================
"""# WHY IT DOESN'T WORK WELL: This plot connects values based on their arbitrary
# index in the test set. Since the order of your test samples has no meaning
# (it's not a time series), the resulting lines are chaotic and uninformative.
# It is included here to demonstrate an anti-pattern in visualization."""

plt.figure(figsize=(12, 6))
# We sort the values by the actual strength to make the line plot less chaotic
# although it's still not the right plot type for this data.
sorted_indices = np.argsort(y_actual)
plt.plot(y_actual[sorted_indices], label='Actual Values (Sorted)', marker='.', linestyle='-', markersize=5)
plt.plot(y_predicted[sorted_indices], label='Predicted Values', marker='.', linestyle='--', markersize=5, alpha=0.7)
plt.title('Line Plot of Predictions (Sorted by Actual Value)', fontsize=16)
plt.xlabel('Sample Index (Sorted by Actual Strength)', fontsize=12)
plt.ylabel('Compressive Strength (MPa)', fontsize=12)
plt.legend()
plt.grid(True)
# plt.savefig('predictions_line_plot.png', dpi=300, bbox_inches='tight')
print("Saved 'predictions_line_plot.png'")


# =========================================================================
# PLOT 3: Histogram of Residuals (Prediction Errors) - A Key Diagnostic Plot
# =========================================================================
"""# WHY IT WORKS: This plot is crucial. The "residuals" are the errors
# (actual - predicted). For a good, unbiased model, this histogram should look
# like a normal distribution (a "bell curve") centered at zero."""

residuals = y_actual - y_predicted
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
plt.title('Histogram of Prediction Errors', fontsize=16)
plt.xlabel('Error (Actual - Predicted) in MPa', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True)
# plt.savefig('residuals_histogram.png', dpi=300, bbox_inches='tight')
print("Saved 'residuals_histogram.png'")
# ==============================================================================
#                 --- ADVANCED PLOTTING AND ANALYSIS ---
#               Place this block with your other plotting code
# ==============================================================================

# --- Step 1: Create a DataFrame for easier analysis ---
# We already have y_actual and y_predicted as NumPy arrays.
df_results = pd.DataFrame({
    'Actual': y_actual,
    'Predicted': y_predicted
})
# =========================================================================
# PLOT 6 (NEW): Residuals vs. Predicted Values
# =========================================================================
"""# WHY IT'S USEFUL: Checks if the error is consistent across all prediction
# levels. A "funnel" shape indicates a problem (heteroscedasticity)."""

residuals = y_actual - y_predicted

plt.figure(figsize=(10, 6))
plt.scatter(y_predicted, residuals, alpha=0.6, edgecolors='k')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.title('Errors vs. Predicted Values', fontsize=16)
plt.xlabel('Predicted Compressive Strength (MPa)', fontsize=12)
plt.ylabel('Errors (Actual - Predicted)', fontsize=12)
plt.grid(True)
# plt.savefig('residuals_vs_predicted.png', dpi=300, bbox_inches='tight')
print("Saved 'residuals_vs_predicted.png'")

