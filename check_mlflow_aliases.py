# check_mlflow_aliases.py (Corrected URI)
import mlflow
import os

# --- 1. Diagnostics ---
print("--- Path and Environment Diagnostics ---")
CWD = os.getcwd()
print(f"Current Working Directory: {CWD}")

db_path = os.path.join(CWD, "mlflow.db")
print(f"Absolute path to mlflow.db being used: {db_path}")
print(f"Does the file exist at this path? -> {os.path.exists(db_path)}")

env_uri = os.getenv("MLFLOW_TRACKING_URI")
print(f"Value of MLFLOW_TRACKING_URI environment variable: {env_uri}")
print("--------------------------------------\n")


# --- 2. More Robust Connection ---
# --- FIX: Corrected SQLite URI from four slashes to three ---
MLFLOW_TRACKING_URI = f"sqlite:///{db_path}"
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "intent-classifier-svc")

print(f"Connecting to MLflow at: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = mlflow.MlflowClient()

print(f"\n--- Checking Registered Model: '{MODEL_NAME}' ---")

try:
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    if not versions:
        print(f"\nERROR: No model with the name '{MODEL_NAME}' was found in the registry.")
    else:
        print(f"\nFound {len(versions)} version(s) for model '{MODEL_NAME}':")
        for v in versions:
            aliases_str = ", ".join(v.aliases) if v.aliases else "No Aliases Set"
            print(f"  - Version: {v.version}, Aliases: [{aliases_str}], Status: {v.status}")

    print("\n" + "="*40)
    print("Verification:")
    print("If you see '[Production]' in the 'Aliases' list for any version, the server will work.")
    print("="*40)

except Exception as e:
    print(f"\nAn error occurred while trying to connect to MLflow: {e}")