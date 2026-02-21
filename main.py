from src.preprocessing import load_and_clean
from src.feature_engineering import select_features
from src.models import build_models, save_model
from src.evaluate import cross_validate_all
from config import DATA_RAW, TARGET

def main():
    print("1. Loading & preprocessing...")
    df = load_and_clean(DATA_RAW)

    print("2. Feature engineering...")
    X, y, selected_features = select_features(df, TARGET)

    print("3. Training & evaluating...")
    models = build_models()
    results = cross_validate_all(models, X, y)
    for name, metrics in results.items():
        print(f"  {name}: RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.3f}")

    print("4. Saving models...")
    for name, model in models.items():
        model.fit(X, y)
        save_model(model, name)

    print("Done!")

if __name__ == "__main__":
    main()