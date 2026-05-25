from data_loader import load_data
from preprocessing import preprocess_data
from feature_engineering import create_features
from eda import run_eda
from modeling import run_modeling
from insights import run_insights

raw_df      = load_data("data/Flight_Price_Dataset_of_Bangladesh.csv")

run_eda(raw_df)

clean_df    = preprocess_data(raw_df)
featured_df = create_features(clean_df)

comparison  = run_modeling(featured_df)
print("\n=== Final Model Comparison ===")
print(comparison.to_string())

run_insights(raw_df, featured_df, comparison)
