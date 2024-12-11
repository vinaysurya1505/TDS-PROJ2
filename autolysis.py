# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "requests",
#   "openai",
#   "scikit-learn",
#   "tabulate",
# ]
# ///
#abdulhadisu
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from datetime import datetime
from sklearn.impute import SimpleImputer



# Ensure the script is run with a CSV filename argument
if len(sys.argv) != 2:
    print("Usage: uv run autolysis.py <dataset.csv>")
    sys.exit(1)

csv_file = sys.argv[1]

if not os.path.exists(csv_file):
    print(f"Error: File '{csv_file}' not found.")
    sys.exit(1)

# Set up the AIPROXY_TOKEN environment variable
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

# Load the dataset with encoding handling
try:
    df = pd.read_csv(csv_file, encoding="utf-8")
except UnicodeDecodeError:
    print("UTF-8 decoding failed. Attempting with 'latin-1' encoding.")
    try:
        df = pd.read_csv(csv_file, encoding="latin-1")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

print(f"Loaded dataset '{csv_file}' with shape {df.shape}")

# Get dataset name without extension
dataset_name = os.path.splitext(os.path.basename(csv_file))[0]

# Create output directory for the dataset
output_dir = os.path.join(os.getcwd(), dataset_name)
os.makedirs(output_dir, exist_ok=True)

# Variables to store results for README
outlier_results = []
feature_importance_results = None
pairplot_result = None
clustering_summary = None
distribution_results = []
trend_analysis_result = None
visualizations = {}
story = None
recommendations = "No specific recommendations were generated."
conclusions = "No specific conclusions were drawn."


def basic_analysis(data):
    """Perform basic analysis on the dataset."""
    analysis = {
        "head": data.head(5).to_dict(orient="records"),
        "columns": list(data.columns),
        "description": data.describe(include="all").to_dict(),
        "missing_values": data.isnull().sum().to_dict()
    }
    return analysis

def ask_llm(prompt):
    """Send a prompt to the LLM via AIProxy and return the response."""
    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data analysis assistant."},
            {"role": "user", "content": prompt},
        ],
    }
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Error: LLM request failed with status {response.status_code}")
        print(response.text)
        sys.exit(1)
 
def suggest_analysis(data_summary):
    """Suggest additional analyses dynamically using LLM."""
    prompt = f"""
    Dataset Summary:
    - Columns: {data_summary['columns']}
    - Missing Values: {data_summary['missing_values']}
    - Key Statistics: {data_summary['description']}
    
    Suggest analyses that would yield meaningful insights from this dataset.
    """
    return ask_llm(prompt)
      
def narrate_analysis(data_summary, analyses):
    """Generate a cohesive story using LLM."""
    global story, recommendations, conclusions
    prompt = f"""
    Dataset Analysis Summary:
    - Columns: {data_summary['columns']}
    - Missing Values: {data_summary['missing_values']}
    - Key Statistics: {data_summary['description']}

    Additional Analysis Results:
    {analyses}

    Provide:
    - Key Insights
    - Dataset Overview
    - Key Findings
    - Recommendations
    - Conclusions
    """
    story = ask_llm(prompt)
    return story

def generate_visualizations(data, output_prefix):
    """Generate visualizations for the dataset."""
    paths = {}

    # Correlation Heatmap
    numeric_cols = data.select_dtypes(include="number").columns
    if not numeric_cols.empty:
        corr = data[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        heatmap_path = f"{output_prefix}/correlation_heatmap.png"
        plt.title("Correlation Heatmap")
        plt.savefig(heatmap_path)
        plt.close()
        paths["heatmap"] = heatmap_path

    # Histograms for numeric columns
    paths["histograms"] = []
    for column in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column], kde=True, color="blue")
        plt.title(f"Histogram of {column}")
        hist_path = f"{output_prefix}/{column}_histogram.png"
        plt.savefig(hist_path)
        plt.close()
        paths["histograms"].append(hist_path)

    return paths

def detect_outliers(data, column):
    """Detect outliers in a numeric column using the IQR method."""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    result = f"Detected outliers in '{column}': {len(outliers)} rows"
    print(result)
    outlier_results.append(result)
    return outliers

def feature_importance_analysis(data, target_column):
    """Analyze feature importance using a Random Forest."""
    global feature_importance_results
    numeric_data = data.select_dtypes(include="number").dropna()
    if target_column not in numeric_data.columns:
        print(f"Target column '{target_column}' is not numeric or not available.")
        return None

    X = numeric_data.drop(columns=[target_column])
    y = numeric_data[target_column]
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("Feature Importance Analysis:")
    print(importance)
    feature_importance_results = importance
    return importance

def correlation_analysis(data):
    """Perform correlation analysis between numerical features."""
    corr = data.corr(numeric_only=True)
    print("Correlation Analysis:")
    print(corr)
    
    # Generate and save correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    corr_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.title("Correlation Matrix")
    plt.savefig(corr_path)
    plt.close()
    
    return corr_path

def pairwise_relationships(data, output_prefix):
    """Generate pairplot for numeric columns."""
    global pairplot_result
    numeric_cols = data.select_dtypes(include="number").columns
    if len(numeric_cols) > 1:
        pairplot_path = f"{output_prefix}/pairplot.png"
        sns.pairplot(data[numeric_cols])
        plt.savefig(pairplot_path)
        plt.close()
        print(f"Pairplot saved as {pairplot_path}")
        pairplot_result = f"Pairplot generated showing relationships between {list(numeric_cols)}."
    else:
        print("Not enough numeric columns for pairplot.")

def clustering_analysis(data, output_prefix, n_clusters=3):
    """Perform K-Means clustering analysis dynamically."""
    global clustering_summary
    numeric_cols = data.select_dtypes(include="number").columns
    if len(numeric_cols) > 1:
        imputer = SimpleImputer(strategy="mean")
        clean_data = pd.DataFrame(imputer.fit_transform(data[numeric_cols]), columns=numeric_cols)
        if clean_data.empty:
            print("All rows have NaN values in numeric columns. Skipping clustering analysis.")
            return

        try:
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clean_data["Cluster"] = kmeans.fit_predict(clean_data)

            # Plot the first two dimensions of the clustering
            plt.figure(figsize=(10, 8))
            sns.scatterplot(
                x=clean_data.iloc[:, 0], y=clean_data.iloc[:, 1], 
                hue=clean_data["Cluster"], palette="viridis", alpha=0.6
            )
            cluster_path = f"{output_prefix}/clustering_scatterplot.png"
            plt.title(f"K-Means Clustering ({n_clusters} Clusters)")
            plt.xlabel(clean_data.columns[0])
            plt.ylabel(clean_data.columns[1])
            plt.savefig(cluster_path)
            plt.close()

            print(f"Clustering scatterplot saved as {cluster_path}")
            clustering_summary = (
                f"K-Means clustering successfully performed with {n_clusters} clusters "
                f"on numeric columns: {list(numeric_cols)}. Results plotted in two dimensions."
            )
        except Exception as e:
            print(f"Error in clustering analysis: {e}")
            clustering_summary = "Clustering analysis failed due to insufficient data or another issue."
    else:
        print("Not enough numeric columns for clustering analysis.")
        clustering_summary = "Clustering analysis was skipped due to insufficient numeric columns."



def distribution_analysis(data, output_prefix):
    """Generate boxplots for numeric columns."""
    global distribution_results
    numeric_cols = data.select_dtypes(include="number").columns
    for column in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[column], color="skyblue")
        plt.title(f"Boxplot of {column}")
        boxplot_path = f"{output_prefix}/{column}_boxplot.png"
        plt.savefig(boxplot_path)
        plt.close()
        print(f"Boxplot for {column} saved as {boxplot_path}")
        distribution_results.append(f"Boxplot created for {column}.")



# Main script logic
if __name__ == "__main__":
    print(f"Performing analysis for dataset '{dataset_name}'...")

    # Perform basic analysis
    data_summary = basic_analysis(df)

    # Detect outliers in numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    for column in numeric_cols:
        detect_outliers(df, column)

    # Perform feature importance analysis (example: using the last numeric column as target)
    if len(numeric_cols) > 1:
        target_column = numeric_cols[-1]
        feature_importance_analysis(df, target_column)

    # Perform pairplot
    pairwise_relationships(df, output_dir)

    # Perform clustering analysis
    clustering_analysis(df, output_dir)

    # Perform distribution analysis
    distribution_analysis(df, output_dir)

    # Perform correlation analysis
    correlation_path = correlation_analysis(df)

    # Generate visualizations
    visualizations = generate_visualizations(df, output_dir)

    # Generate narrative using LLM
    analyses_summary = f"""
    Outlier Detection: {outlier_results}
    Feature Importance: {feature_importance_results.to_dict() if feature_importance_results is not None else "N/A"}
    Clustering: {clustering_summary if clustering_summary else "N/A"}
    """
    narrate_analysis(data_summary, analyses_summary)

    # Generate analysis suggestions from LLM
    suggestions = suggest_analysis(data_summary)

    # Write results to README.md
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# Automated Data Analysis Report for {dataset_name.capitalize()}\n\n")
        f.write(f"## Dataset: {csv_file}\n\n")
        f.write("### Dataset Overview\n")
        f.write(f"- **Columns**: {data_summary['columns']}\n")
        f.write(f"- **Missing Values**: {data_summary['missing_values']}\n\n")
        f.write(story + "\n\n")
        f.write("### Outlier Detection Results\n")
        f.write("\n".join(outlier_results) + "\n\n")
        if feature_importance_results is not None:
            f.write("### Feature Importance Analysis\n")
            f.write(feature_importance_results.to_markdown() + "\n\n")
        f.write(f"### Correlation Analysis\n")
        f.write(f"Correlation Matrix saved as {correlation_path}\n\n")
        if clustering_summary:
            f.write(f"### Clustering Analysis\n")
            f.write(clustering_summary + "\n\n")
        if distribution_results:
            f.write("### Distribution Analysis\n")
            f.write("\n".join(distribution_results) + "\n\n")
        f.write("### Visualizations\n")
        if "heatmap" in visualizations:
            f.write(f"![Correlation Heatmap]({visualizations['heatmap']})\n")
        for hist in visualizations["histograms"]:
            f.write(f"![Histogram: {hist}]({os.path.basename(hist)})\n")
        f.write("### Suggestions\n")
        f.write(suggestions + "\n")
    print(f"Analysis completed. Results saved in '{output_dir}'.")