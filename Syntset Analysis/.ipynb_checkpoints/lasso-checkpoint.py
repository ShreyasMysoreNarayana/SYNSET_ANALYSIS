import argparse
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from lda import apply_lda
from collections import Counter

import argparse
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def correlation_thresholding(df, threshold=0.9):
    """
    Removes highly correlated features based on a given threshold.
    :param df: DataFrame containing the features
    :param threshold: Threshold above which features are considered highly correlated
    :return: DataFrame with reduced features
    """
    corr_matrix = df.corr().abs()  # Compute absolute correlation matrix
    upper_tri = corr_matrix.where(pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool))  # Select upper triangle
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]  # Find columns to drop
    return df.drop(columns=to_drop)

def main():
    parser = argparse.ArgumentParser(description='Process feature selection and evaluation.')
    parser.add_argument('input_file', type=str, help='Input CSV file for processing')
    parser.add_argument('output_file', type=str, help='Output file to save selected features')
    parser.add_argument('--var_thresh', type=float, default=0.0, help='Variance threshold (default: 0.0)')
    parser.add_argument('--corr_thresh', type=float, default=0.9, help='Correlation threshold (default: 0.9)')
    args = parser.parse_args()

    # Read the input CSV file
    result_df = pd.read_csv(args.input_file)
    
    # Separate features and target
    y = result_df['decision'].astype(bool)
    X = result_df.drop(columns=['Subject', 'decision', 'neuropsych_score'], axis=1)

    # Step 1: Variance Thresholding
    print(f"Applying variance threshold: {args.var_thresh}")
    selector = VarianceThreshold(threshold=args.var_thresh)
    X_var_filtered = selector.fit_transform(X)
    X_var_filtered_df = pd.DataFrame(X_var_filtered, columns=X.columns[selector.get_support()])

    # Step 2: Correlation Thresholding
    print(f"Applying correlation threshold: {args.corr_thresh}")
    numeric_df = correlation_thresholding(X_var_filtered_df, threshold=args.corr_thresh)

    # Perform Lasso regression for feature selection
    lasso = Lasso(alpha=0.03)  # You can adjust the alpha value
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    lasso.fit(X_scaled, y)

    # Get the selected features
    selected_features = numeric_df.columns[lasso.coef_ != 0]
    selected_features_df = numeric_df[selected_features]

    # Save the selected features to the output file
    selected_features_df.to_csv(args.output_file, index=False)

    print(f"Selected {len(selected_features)} features out of {numeric_df.shape[1]}")

    apply_lda(numeric_df[selected_features], y, result_df['neuropsych_score'])
if __name__ == "__main__":
    main()

