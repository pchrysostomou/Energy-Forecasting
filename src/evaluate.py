"""
Evaluation & Visualisation Module

Owners: Hewen & Prodomos (Team C)

This module implements all metrics and plotting functions.

=== METRICS ===

1. RMSE (Root Mean Squared Error) — point prediction quality
   - Applies to ALL models
   - For Gaussian: use mu as point prediction
   - For MC Dropout: use mean prediction

2. NLL (Negative Log-Likelihood) — probabilistic forecast quality
   - Applies to Gaussian and MC Dropout only
   - For MC Dropout: fit a Gaussian to the T predictions (mean, std)

3. CRPS (Continuous Ranked Probability Score) — proper scoring rule
   - Like MAE but for distributions
   - Use `properscoring` library: pip install properscoring
   - For Gaussian: properscoring.crps_gaussian(y_true, mu, sigma)

4. Calibration Error — are confidence intervals honest?
   - Compute prediction intervals at levels [10%, 20%, ..., 90%]
   - Check what fraction of true values fall inside each interval
   - Perfect calibration: 90% interval contains exactly 90% of truths

5. Sharpness — average width of 90% prediction interval
   - Narrower is better (if well-calibrated)

=== PLOTS ===

1. Prediction plot: time series of true values + model predictions + shaded uncertainty
2. Calibration plot: expected coverage (x) vs observed coverage (y), diagonal = perfect
3. Ablation table: compare all models side by side on all metrics
4. Training curves: loss vs epoch for each model

=== TODO (Hewen & Prodomos) ===

Hewen:
    1. Implement compute_rmse(), compute_nll(), compute_crps()
    2. Implement compute_calibration() — returns expected and observed coverages
    3. Test each metric on synthetic data first (e.g., known Gaussian)

Prodomos:
    1. Implement plot_predictions_with_uncertainty()
    2. Implement plot_calibration()
    3. Implement plot_training_curves()
    4. All figures should be publication-quality (labelled axes, legend, title)
    5. Save all figures to figures/ directory

=== USAGE EXAMPLE ===

    from src.evaluate import compute_rmse, compute_crps, plot_calibration

    rmse = compute_rmse(y_true, y_pred)
    crps = compute_crps(y_true, mu, sigma)
    plot_calibration(y_true, mu, sigma, save_path="figures/calibration.png")
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_nll(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    Gaussian Negative Log-Likelihood.

    NLL = 0.5 * log(σ²) + (y - μ)² / (2σ²)
    """
    variance = sigma ** 2
    nll = 0.5 * np.log(variance) + (y_true - mu) ** 2 / (2 * variance)
    return np.mean(nll)


def compute_crps(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Continuous Ranked Probability Score for Gaussian predictions.

    Closed-form CRPS for a Gaussian forecast (Gneiting & Raftery, 2007):

        CRPS(N(μ, σ²); y) = σ * [ z*(2*Φ(z) - 1) + 2*φ(z) - 1/√π ]

    where z = (y - μ) / σ, Φ is the standard normal CDF and φ its PDF.
    Lower is better. Inlined here so we don't need the `properscoring`
    package — keeps the additional-package count to three.
    """
    from scipy.stats import norm

    sigma = np.maximum(sigma, 1e-12)  # avoid division by zero
    z = (y_true - mu) / sigma
    crps = sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def compute_calibration(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                         levels: np.ndarray = None) -> dict:
    """Compute calibration: observed vs expected coverage for prediction intervals.

    For each confidence level p, compute the symmetric prediction interval
    [mu - z*sigma, mu + z*sigma] where z = norm.ppf((1+p)/2), then count
    what fraction of true values actually fall inside.

    Args:
        y_true: (N,) true values.
        mu: (N,) predicted means.
        sigma: (N,) predicted stds.
        levels: Confidence levels to check (default: 0.1, 0.2, ..., 0.9).

    Returns:
        dict with "expected" and "observed" lists (same length as levels).
    """
    from scipy.stats import norm

    if levels is None:
        levels = np.arange(0.1, 1.0, 0.1)

    observed = []
    for p in levels:
        z = norm.ppf((1 + p) / 2)  # e.g. p=0.9 -> z=1.645
        lower = mu - z * sigma
        upper = mu + z * sigma
        frac = np.mean((y_true >= lower) & (y_true <= upper))
        observed.append(float(frac))

    return {"expected": [float(p) for p in levels], "observed": observed}


def find_temperature(y_true, mu, sigma):
    """Find the temperature scalar T that minimises NLL on a held-out set.

    Temperature scaling recalibrates a Gaussian model by multiplying the
    predicted sigma by a single scalar T:  sigma_calibrated = sigma_raw * T.
    T > 1 widens the intervals (fixes overconfidence).

    Uses scipy.optimize.minimize_scalar for a clean, fast solution.

    Args:
        y_true: (N,) true values (normalised).
        mu:     (N,) predicted means (normalised).
        sigma:  (N,) predicted stds (normalised, raw from model).

    Returns:
        (T_opt, T_grid, nll_grid) where T_opt is the optimal scalar,
        and T_grid/nll_grid are arrays for plotting the search landscape.
    """
    from scipy.optimize import minimize_scalar

    def nll_at_T(T):
        s = sigma * T
        var = s ** 2
        return float(np.mean(0.5 * np.log(var) + (y_true - mu) ** 2 / (2 * var)))

    result = minimize_scalar(nll_at_T, bounds=(0.1, 50.0), method="bounded")
    T_opt = float(result.x)

    # Also build a grid for the search landscape plot
    T_grid = np.linspace(0.5, min(T_opt * 2.5, 50.0), 500)
    nll_grid = np.array([nll_at_T(t) for t in T_grid])

    return T_opt, T_grid, nll_grid


def plot_predictions(y_true, y_pred, save_path=None, title="",
                     sigma=None):
    """Plot true demand vs predicted demand over a time window.

    When *sigma* is provided (probabilistic models), shaded 95% and 68%
    confidence bands are drawn around the prediction.

    Args:
        y_true: (N,) array of true values in original units (MW).
        y_pred: (N,) array of predicted values (or mu for probabilistic).
        save_path: If given, figure is saved to this path.
        title: Plot title.
        sigma: (N,) optional predicted std for uncertainty bands.
    """
    hours = np.arange(len(y_true)) * 0.5  # half-hourly -> hours

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hours, y_true, label="Actual demand", color="#1f77b4", linewidth=1.2)
    ax.plot(hours, y_pred, label="Predicted demand", color="#ff7f0e",
            linewidth=1.2, linestyle="--")

    if sigma is not None:
        sigma = np.asarray(sigma)
        ax.fill_between(hours, y_pred - 1.96 * sigma, y_pred + 1.96 * sigma,
                         alpha=0.15, color="#ff7f0e", label="95% CI")
        ax.fill_between(hours, y_pred - sigma, y_pred + sigma,
                         alpha=0.30, color="#ff7f0e", label="68% CI")

    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("Demand (MW)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_predictions_with_uncertainty(y_true, mu, sigma, n_steps=200,
                                       save_path=None, title=""):
    """Convenience wrapper kept for backward compatibility."""
    plot_predictions(y_true[:n_steps], mu[:n_steps], save_path=save_path,
                     title=title, sigma=sigma[:n_steps] if sigma is not None else None)


def plot_calibration(expected, observed, save_path=None, title=""):
    """Calibration plot: expected coverage vs observed coverage.

    Points on the diagonal indicate perfect calibration. Below the diagonal
    means the model is overconfident; above means underconfident.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.plot(expected, observed, "o-", color="#d62728", linewidth=1.5,
            markersize=6, label="Model calibration")
    ax.fill_between(expected, expected, observed, alpha=0.15, color="#d62728")

    cal_error = np.mean(np.abs(np.array(expected) - np.array(observed)))
    ax.text(0.05, 0.90, f"Cal. Error: {cal_error:.4f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Expected coverage", fontsize=12)
    ax.set_ylabel("Observed coverage", fontsize=12)
    ax.set_title(title or "Calibration Plot", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_sigma_analysis(y_true, sigma, save_path=None, title=""):
    """Plot predicted sigma alongside actual demand to visualise uncertainty.

    Overlays de-normalised sigma on a secondary y-axis against the true
    demand signal, so the viewer can see whether the model outputs higher
    uncertainty during volatile periods.
    """
    hours = np.arange(len(y_true)) * 0.5

    fig, ax1 = plt.subplots(figsize=(14, 5))

    colour_demand = "#1f77b4"
    colour_sigma = "#d62728"

    ax1.plot(hours, y_true, color=colour_demand, linewidth=1.2, label="Actual demand")
    ax1.set_xlabel("Time (hours)", fontsize=12)
    ax1.set_ylabel("Demand (MW)", fontsize=12, color=colour_demand)
    ax1.tick_params(axis="y", labelcolor=colour_demand)

    ax2 = ax1.twinx()
    ax2.plot(hours, sigma, color=colour_sigma, linewidth=1.2, alpha=0.8,
             label="Predicted sigma")
    ax2.set_ylabel("Predicted sigma (MW)", fontsize=12, color=colour_sigma)
    ax2.tick_params(axis="y", labelcolor=colour_sigma)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left")

    ax1.set_title(title or "Sigma Analysis", fontsize=13)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_training_curves(train_losses, val_losses, save_path=None, title="",
                         ylabel="Loss"):
    """Plot training and validation loss curves.

    Args:
        train_losses: List of per-epoch training losses.
        val_losses: List of per-epoch validation losses.
        save_path: If given, figure is saved to this path.
        title: Plot title.
        ylabel: Label for the y-axis (e.g. "Loss (MSE)" or "Loss (NLL)").
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train loss", color="#1f77b4",
            linewidth=1.5, marker="o", markersize=4)
    ax.plot(epochs, val_losses, label="Validation loss", color="#ff7f0e",
            linewidth=1.5, marker="s", markersize=4)

    best_epoch = int(np.argmin(val_losses)) + 1
    best_val = min(val_losses)
    ax.axvline(best_epoch, color="grey", linestyle=":", alpha=0.6)
    ax.annotate(f"Best epoch {best_epoch}\n(val={best_val:.4f})",
                xy=(best_epoch, best_val),
                xytext=(best_epoch + 1, best_val + 0.05),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="grey"))

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
