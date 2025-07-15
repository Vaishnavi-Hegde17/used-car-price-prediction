import mlflow
import mlflow.sklearn
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ------------------- Load Params -------------------
with open("configs/params.yaml") as f:
    params = yaml.safe_load(f)

test_size = params["data_split"]["test_size"]
random_state = params["data_split"]["random_state"]
n_neighbors = params["models"]["knn"]["n_neighbors"]
n_estimators = params["models"]["rf"]["n_estimators"]
scaling_enabled = params["scaling"]["enabled"]
experiment_name = params["mlflow"]["experiment_name"]

# ------------------- Paths -------------------
raw_data_path = "data/raw/cardata.csv"
processed_path = "data/processed"
models_path = "models"
plots_path = "saved_plots"

os.makedirs(processed_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)
os.makedirs(plots_path, exist_ok=True)

# ------------------- Load Dataset -------------------
df = pd.read_csv(raw_data_path)

# ------------------- Preprocessing -------------------
le = LabelEncoder()
df["Fuel_Type"] = le.fit_transform(df["Fuel_Type"])
df["Seller_Type"] = le.fit_transform(df["Seller_Type"])
df["Transmission"] = le.fit_transform(df["Transmission"])

X = df[["Year", "Present_Price", "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", "Owner"]]
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Scaling (only for KNN)
if scaling_enabled:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
else:
    X_train_scaled = X_train
    X_test_scaled = X_test

# ------------------- Save Processed Data -------------------
X_train.to_csv(os.path.join(processed_path, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(processed_path, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(processed_path, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(processed_path, "y_test.csv"), index=False)

print("\n Preprocessing Complete. Data saved in data/processed/\n")

# ------------------- Start MLflow Experiment -------------------
mlflow.set_experiment(experiment_name)

# ------------------- Linear Regression -------------------
with mlflow.start_run(run_name="Linear Regression"):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)

    mlflow.log_param("model", "Linear Regression")
    mlflow.log_metric("rmse", rmse_lr)
    mlflow.log_metric("r2_score", r2_lr)
    mlflow.sklearn.log_model(lr, "linear_regression_model")

    with open(os.path.join(models_path, "linear_regression.pkl"), "wb") as f:
        pickle.dump(lr, f)

    # Plot
    plt.figure()
    plt.scatter(y_test, y_pred_lr, color='blue', edgecolors='k')
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title("Linear Regression: Actual vs Predicted")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plot_lr = os.path.join(plots_path, "lr_plot.png")
    plt.savefig(plot_lr)
    plt.close()
    mlflow.log_artifact(plot_lr, artifact_path="plots")

    print("\n--- Linear Regression Model Saved ---")
    print(f"RMSE: {rmse_lr:.2f}")
    print(f"R² Score: {r2_lr:.2f}")

# ------------------- KNN Regressor -------------------
with mlflow.start_run(run_name="KNN Regressor") as run:
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)

    rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
    r2_knn = r2_score(y_test, y_pred_knn)

    mlflow.log_param("model", "KNN Regressor")
    mlflow.log_param("n_neighbors", n_neighbors)
    mlflow.log_metric("rmse", rmse_knn)
    mlflow.log_metric("r2_score", r2_knn)
    mlflow.sklearn.log_model(knn, "knn_regressor_model")

    with open(os.path.join(models_path, "knn_regressor.pkl"), "wb") as f:
        pickle.dump(knn, f)

    # Plot
    plt.figure()
    plt.scatter(y_test, y_pred_knn, color='green', edgecolors='k')
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title("KNN Regressor: Actual vs Predicted")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plot_knn = os.path.join(plots_path, "knn_plot.png")
    plt.savefig(plot_knn)
    plt.close()
    mlflow.log_artifact(plot_knn, artifact_path="plots")

    result = mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/knn_regressor_model",
        name="KNNCarPriceRegressor"
    )
    print("\n--- KNN Regressor Model Saved ---")
    print(f"RMSE: {rmse_knn:.2f}")
    print(f"R² Score: {r2_knn:.2f}")
    print(f"Model registered: {result.name}, version: {result.version}")

# ------------------- Random Forest -------------------
with mlflow.start_run(run_name="Random Forest Regressor") as run:
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    mlflow.log_param("model", "Random Forest Regressor")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("rmse", rmse_rf)
    mlflow.log_metric("r2_score", r2_rf)
    mlflow.sklearn.log_model(rf, "random_forest_model")

    with open(os.path.join(models_path, "random_forest.pkl"), "wb") as f:
        pickle.dump(rf, f)

    # Plot
    plt.figure()
    plt.scatter(y_test, y_pred_rf, color='orange', edgecolors='k')
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title("Random Forest: Actual vs Predicted")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plot_rf = os.path.join(plots_path, "rf_plot.png")
    plt.savefig(plot_rf)
    plt.close()
    mlflow.log_artifact(plot_rf, artifact_path="plots")

    result_rf = mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/random_forest_model",
        name="RFCarPriceRegressor"
    )
    print("\n--- Random Forest Model Saved ---")
    print(f"RMSE: {rmse_rf:.2f}")
    print(f"R² Score: {r2_rf:.2f}")
    print(f"Model registered: {result_rf.name}, version: {result_rf.version}")

# ------------------- Summary -------------------
print("\n===== Model Comparison =====")
models_r2 = {
    "Linear Regression": r2_lr,
    "KNN Regressor": r2_knn,
    "Random Forest": r2_rf
}
best_model = max(models_r2, key=models_r2.get)
print(f"✅ Best Model Based on R² Score: {best_model}")
