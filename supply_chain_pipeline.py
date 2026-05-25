"""
Supply Chain Cost Optimization using CNN-based Feature Extraction
=================================================================
Hybrid ML pipeline: ResNet50 image features + metadata → price prediction + anomaly detection
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import json

# ─── Config ───────────────────────────────────────────────────────────────────
STYLES_CSV   = "/mnt/user-data/uploads/styles.csv"
IMAGES_DIR   = "images"           # adjust if you have the images locally
SAMPLE_SIZE  = 8000               # MVP cap
BATCH_SIZE   = 32
EPOCHS       = 8
TEST_SIZE    = 0.2
IMG_SIZE     = (224, 224)
ANOMALY_TOP_PCT = 0.10            # top 10% errors → anomaly
RANDOM_SEED  = 42
np.random.seed(RANDOM_SEED)

# ─── 1. Load & Clean ──────────────────────────────────────────────────────────
print("\n[1/7] Loading & cleaning dataset …")
df = pd.read_csv(STYLES_CSV, on_bad_lines="skip")
df.dropna(subset=["gender","masterCategory","subCategory",
                   "articleType","baseColour","season","usage"], inplace=True)
df.reset_index(drop=True, inplace=True)

# keep only rows where the image file exists (if images dir present)
images_available = Path(IMAGES_DIR).exists()
if images_available:
    df["image_path"] = df["id"].apply(lambda x: f"{IMAGES_DIR}/{x}.jpg")
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"   Rows with images found: {len(df)}")
else:
    print(f"   Images directory '{IMAGES_DIR}' not found — running in metadata-only mode.")
    print(f"   (Place the Kaggle 'images/' folder next to this script to enable CNN features.)")
    df["image_path"] = df["id"].apply(lambda x: f"{IMAGES_DIR}/{x}.jpg")

# Sample for MVP
if len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"   Working with {len(df):,} rows.")

# ─── 2. Synthetic Price Engineering ───────────────────────────────────────────
print("\n[2/7] Engineering synthetic price …")

BASE_PRICE = {
    "Tshirts": (400, 900),    "Shirts": (600, 1500),    "Tops": (350, 900),
    "Kurtas": (500, 2000),    "Jeans": (800, 2500),     "Track Pants": (400, 1200),
    "Casual Shoes": (1000, 3500), "Sports Shoes": (1500, 5000), "Heels": (800, 3000),
    "Sandals": (400, 1800),   "Flip Flops": (200, 700),
    "Watches": (1500, 8000),  "Handbags": (800, 4000),  "Wallets": (400, 1500),
    "Sunglasses": (500, 2500),"Belts": (300, 1200),     "Socks": (100, 400),
    "Jackets": (1500, 6000),  "Sweaters": (800, 3000),  "Bra": (300, 900),
    "Briefs": (150, 500),     "default": (300, 1500)
}

SEASON_MULT   = {"Winter": 1.30, "Fall": 1.15, "Spring": 1.05, "Summer": 0.90}
USAGE_MULT    = {"Sports": 1.25, "Formal": 1.20, "Smart Casual": 1.15,
                 "Party": 1.10,  "Ethnic": 1.05, "Travel": 1.00,
                 "Casual": 0.95, "Home": 0.85}
GENDER_MULT   = {"Women": 1.10, "Men": 1.00, "Boys": 0.85, "Girls": 0.85, "Unisex": 0.95}
CATEGORY_MULT = {"Footwear": 1.20, "Accessories": 1.10, "Apparel": 1.00,
                 "Sporting Goods": 1.15, "Personal Care": 0.80,
                 "Free Items": 0.50, "Home": 0.90}

rng = np.random.default_rng(RANDOM_SEED)

def synthetic_price(row):
    lo, hi   = BASE_PRICE.get(row["articleType"], BASE_PRICE["default"])
    base      = rng.uniform(lo, hi)
    sm        = SEASON_MULT.get(row["season"], 1.0)
    um        = USAGE_MULT.get(row["usage"], 1.0)
    gm        = GENDER_MULT.get(row["gender"], 1.0)
    cm        = CATEGORY_MULT.get(row["masterCategory"], 1.0)
    noise     = rng.uniform(0.92, 1.08)
    return round(base * sm * um * gm * cm * noise, 2)

df["synthetic_price"] = df.apply(synthetic_price, axis=1)
print(f"   Price range: ₹{df['synthetic_price'].min():.0f} – ₹{df['synthetic_price'].max():.0f}")
print(f"   Mean price : ₹{df['synthetic_price'].mean():.0f}")

# ─── 3. Metadata Encoding ─────────────────────────────────────────────────────
print("\n[3/7] Encoding metadata features …")
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

CAT_COLS = ["gender","masterCategory","subCategory","articleType",
            "baseColour","season","usage"]

encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

meta_features = [f"{c}_enc" for c in CAT_COLS]
X_meta = df[meta_features].values.astype(np.float32)
y      = df["synthetic_price"].values.astype(np.float32)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# ─── 4. CNN Feature Extraction (if images available) ─────────────────────────
print("\n[4/7] CNN feature extraction …")

if images_available:
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input
        from tensorflow.keras.preprocessing import image as kimage

        print("   Loading ResNet50 (pretrained on ImageNet) …")
        base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg",
                              input_shape=(224, 224, 3))
        base_model.trainable = False

        def extract_features_batch(paths, bsz=32):
            feats = []
            for i in range(0, len(paths), bsz):
                batch_paths = paths[i:i+bsz]
                imgs = []
                for p in batch_paths:
                    img = kimage.load_img(p, target_size=IMG_SIZE)
                    arr = kimage.img_to_array(img)
                    imgs.append(arr)
                batch = preprocess_input(np.stack(imgs))
                feats.append(base_model.predict(batch, verbose=0))
                if (i // bsz) % 10 == 0:
                    print(f"   Processed {min(i+bsz, len(paths))}/{len(paths)} images …", end="\r")
            return np.vstack(feats)

        img_features = extract_features_batch(df["image_path"].tolist())
        print(f"\n   Image feature shape: {img_features.shape}")
        X_combined = np.hstack([img_features, X_meta])
        print(f"   Combined feature vector: {X_combined.shape[1]} dims")
    except Exception as e:
        print(f"   TensorFlow/image error: {e}")
        print("   → Falling back to metadata-only mode.")
        images_available = False
        X_combined = X_meta

else:
    print("   No images found — using metadata features only.")
    X_combined = X_meta

# ─── 5. Train / Test Split ────────────────────────────────────────────────────
print("\n[5/7] Splitting data & training model …")
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_combined, y_scaled, df.index.values,
    test_size=TEST_SIZE, random_state=RANDOM_SEED
)

scaler_X = StandardScaler()
X_train  = scaler_X.fit_transform(X_train)
X_test   = scaler_X.transform(X_test)

# ─── 6. Price Prediction Model ────────────────────────────────────────────────
# Primary: Neural Network via scikit-learn MLPRegressor (no TF dependency required)
# Falls back to XGBoost if available

USE_NN = True
try:
    from sklearn.neural_network import MLPRegressor
    print("   Training MLP Regressor (Dense 512→128→1) …")
    model = MLPRegressor(
        hidden_layer_sizes=(512, 128),
        activation="relu",
        solver="adam",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_SEED,
        verbose=False
    )
    model.fit(X_train, y_train)
    print("   MLP training complete.")
except Exception as e:
    print(f"   MLP failed ({e}), trying XGBoost …")
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=300, learning_rate=0.05,
                         max_depth=6, random_state=RANDOM_SEED, verbosity=0)
    model.fit(X_train, y_train)
    print("   XGBoost training complete.")

# ─── Evaluate ─────────────────────────────────────────────────────────────────
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
print(f"\n   ── Evaluation Metrics ──────────────────")
print(f"   RMSE : ₹{rmse:,.2f}")
print(f"   MAE  : ₹{mae:,.2f}")
print(f"   MAPE : {np.mean(np.abs((y_true - y_pred)/y_true))*100:.2f}%")

# ─── 7. Anomaly Detection ─────────────────────────────────────────────────────
print("\n[6/7] Running anomaly detection …")

errors   = np.abs(y_true - y_pred)
threshold = np.percentile(errors, (1 - ANOMALY_TOP_PCT) * 100)
print(f"   Anomaly threshold (top {ANOMALY_TOP_PCT*100:.0f}% error): ₹{threshold:,.2f}")

test_df = df.loc[idx_test].copy()
test_df["actual_price"]    = y_true
test_df["predicted_price"] = np.round(y_pred, 2)
test_df["error"]           = np.round(errors, 2)

def status(row):
    if row["error"] <= threshold:
        return "Normal"
    return "Overpriced" if row["actual_price"] > row["predicted_price"] else "Underpriced"

test_df["status"] = test_df.apply(status, axis=1)

summary = test_df["status"].value_counts()
print(f"\n   Status breakdown:\n{summary.to_string()}")

# ─── 8. Save Outputs ──────────────────────────────────────────────────────────
print("\n[7/7] Saving outputs …")
OUT_DIR = Path("/mnt/user-data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Full results table
results_cols = ["id","gender","masterCategory","articleType",
                "season","usage","actual_price","predicted_price","error","status"]
results_df = test_df[results_cols].sort_values("error", ascending=False)
results_df.to_csv(OUT_DIR / "anomaly_detection_results.csv", index=False)
print(f"   → anomaly_detection_results.csv  ({len(results_df):,} rows)")

# Top 10 anomalies
top10 = results_df[results_df["status"] != "Normal"].head(10)
top10.to_csv(OUT_DIR / "top10_anomalies.csv", index=False)
print(f"   → top10_anomalies.csv")

# Metrics JSON (for the dashboard)
metrics = {
    "rmse": round(float(rmse), 2),
    "mae":  round(float(mae), 2),
    "mape": round(float(np.mean(np.abs((y_true - y_pred)/y_true))*100), 2),
    "threshold": round(float(threshold), 2),
    "n_test": int(len(results_df)),
    "status_counts": summary.to_dict(),
    "mode": "CNN + Metadata" if images_available else "Metadata Only",
    "sample_size": SAMPLE_SIZE,
}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"   → metrics.json")

# Price distribution sample (for dashboard charts)
sample_chart = results_df[["actual_price","predicted_price","error","status"]].head(200)
sample_chart.to_csv(OUT_DIR / "chart_data.csv", index=False)
print(f"   → chart_data.csv")

print("\n✅  Pipeline complete!\n")
print("=" * 55)
print(f"  Mode         : {metrics['mode']}")
print(f"  Samples      : {SAMPLE_SIZE:,}")
print(f"  RMSE         : ₹{rmse:,.2f}")
print(f"  MAE          : ₹{mae:,.2f}")
print(f"  Anomalies    : {summary.get('Overpriced',0) + summary.get('Underpriced',0)}")
print(f"  Normal       : {summary.get('Normal',0)}")
print("=" * 55)
print()
print("Output files in /mnt/user-data/outputs/:")
print("  • anomaly_detection_results.csv")
print("  • top10_anomalies.csv")
print("  • metrics.json")
print("  • chart_data.csv")
