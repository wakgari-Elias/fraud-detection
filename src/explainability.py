# src/explainability.py
from __future__ import annotations

import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from typing import Any, List, Optional, Tuple
from pathlib import Path
import warnings

# SHAP kmeans summarizer
from shap import kmeans

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)  # sometimes from matplotlib/shap


def load_model(model_path: str) -> Any:
    """
    Load a saved model from disk (supports joblib/pickle).

    Args:
        model_path: Path to the saved model file (.joblib recommended)

    Returns:
        Loaded model object

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = joblib.load(path)
    print(f"Loaded model from {path}")
    return model


def get_feature_importance(
    model: Any,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract built-in feature importance from tree-based models (XGBoost, RandomForest, etc.).

    Args:
        model: Trained tree-based model
        feature_names: List of feature names matching model's input

    Returns:
        Sorted DataFrame with feature and importance scores
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not support feature_importances_ attribute")

    importances = model.feature_importances_
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    return df


def plot_top_features(
    importance_df: pd.DataFrame,
    top_n: int = 10,
    save_path: Optional[str] = "figures/top_features.png"
) -> None:
    """
    Bar plot of top N feature importances.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df.head(top_n),
        x='importance',
        y='feature',
        palette='viridis'
    )
    plt.title(f'Top {top_n} Built-in Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved built-in importance plot → {save_path}")

    plt.show()


def explain_with_shap(
    model: Any,
    X: pd.DataFrame,
    max_display: int = 15,
    sample_size: int = 200,               # lowered default to avoid crashes
    save_dir: str = "figures/shap",
    use_kmeans_background: bool = True,   # ← key anti-crash setting
    kmeans_clusters: int = 30             # good balance of speed/accuracy
) -> Tuple[shap.Explanation, pd.DataFrame]:  # Updated return: both explanation and X_sample
    """
    Compute SHAP values using TreeExplainer + generate summary plots.
    Uses kmeans background summarization by default to prevent OOM/kernel crashes.

    Args:
        model: Trained XGBoost (or compatible tree) model
        X: Feature DataFrame
        max_display: Max features in summary plot
        sample_size: Number of instances to explain (small = faster/safer)
        save_dir: Directory to save plots
        use_kmeans_background: If True, summarizes background data (strongly recommended)
        kmeans_clusters: Number of clusters for background summary

    Returns:
        Tuple of SHAP Explanation object and the sampled DataFrame
    """
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    print(f"Preparing SHAP explanation | sample_size={sample_size}, kmeans={use_kmeans_background}")

    # Summarize background if enabled (huge memory saver)
    if use_kmeans_background:
        print(f"Summarizing background data to {kmeans_clusters} clusters...")
        background = kmeans(X, kmeans_clusters).data
    else:
        background = X  # raw — risky for large data

    print("Creating TreeExplainer...")
    explainer = shap.TreeExplainer(model, background)

    # Sample data to explain
    print(f"Sampling {sample_size} instances for explanation...")
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)

    print("Computing SHAP values... (this may take a while)")
    shap_values = explainer.shap_values(X_sample)

    print("Generating summary plots...")

    # Bar summary (global importance)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        max_display=max_display,
        show=False
    )
    plt.title("SHAP Summary — Global Feature Importance (Bar)")
    bar_path = Path(save_dir) / "shap_summary_bar.png"
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()  # avoid display issues
    print(f"Saved bar summary → {bar_path}")

    # Beeswarm (detailed distribution)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=max_display,
        show=False
    )
    plt.title("SHAP Summary — Beeswarm (Feature Impact Distribution)")
    swarm_path = Path(save_dir) / "shap_summary_beeswarm.png"
    plt.savefig(swarm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved beeswarm summary → {swarm_path}")

    print("SHAP computation complete ✓")

    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_sample.values,
        feature_names=X_sample.columns.tolist()
    )
    return explanation, X_sample  # Now return both


def plot_shap_force(
    explanation: shap.Explanation,
    index: int,
    save_path: Optional[str] = None,
    title_suffix: str = ""
) -> None:
    """
    Generate matplotlib force plot for a single instance.
    """
    plt.figure(figsize=(12, 4))
    shap.force_plot(
        explanation.base_values,
        explanation.values[index],
        explanation.data[index],
        feature_names=explanation.feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f"SHAP Force Plot - Instance {index}{title_suffix}")
    
    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved force plot → {save_path}")
    
    plt.show()


def plot_shap_dependence(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature: str,
    interact_feature: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Generate SHAP dependence plot for a feature (capstone enhancement for interactions).
    """
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature,
        shap_values,
        X_sample,
        interaction_index=interact_feature,
        show=False
    )
    plt.title(f"SHAP Dependence Plot: {feature} (Interaction with {interact_feature or 'Auto'})")
    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved dependence plot: {save_path}")
    plt.show()