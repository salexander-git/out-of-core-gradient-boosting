import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler


def main():
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    scaler = MinMaxScaler(feature_range=(-10.0, 10.0))
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    y = y.astype(np.float32)

    X_scaled.tofile("california_features.bin")
    y.tofile("california_labels.bin")

    f_min, f_max = -10.0, 10.0
    scale = 255.0 / (f_max - f_min)
    X_qbin = np.clip(((X_scaled - f_min) * scale), 0, 255).astype(np.uint8)
    X_qbin.tofile("california_features_qbin.bin")

    print("Wrote:")
    print("  california_features.bin")
    print("  california_features_qbin.bin")
    print("  california_labels.bin")
    print(f"Rows: {X_scaled.shape[0]}, Features: {X_scaled.shape[1]}")


if __name__ == "__main__":
    main()