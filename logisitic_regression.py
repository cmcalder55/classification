import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from typing import Tuple, List, Optional

from utils.plotter import FloatArray


class PolynomialFitter:
    """
    Class for generating synthetic noisy polynomial data and fitting a polynomial model.
    """

    def __init__(
        self,
        order: int = 3,
        n_samples: int = 50,
        noise_scale: float = 100,
        seed: int = 42,
    ):
        self.order = order
        self.n_samples = n_samples
        self.noise_scale = noise_scale
        self.seed = seed

        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.poly: Optional[np.poly1d] = None
        self.t: Optional[np.ndarray] = None

    def generate_data(self, x_range: Tuple[float, float] = (-15, 15)):
        """
        Generate synthetic noisy polynomial data.

        Parameters:
            x_range: The range of x values to sample.

        Returns:
            A tuple of (x, y) data arrays.
        """
        np.random.seed(self.seed)
        self.x = 30 * (np.random.rand(self.n_samples) - 0.5)
        true_y = 5 * self.x + 15 * self.x**2 + 2 * self.x**3
        noise = self.noise_scale * np.random.randn(self.n_samples)
        self.y = true_y + noise
        self.t = np.linspace(*x_range, self.n_samples)

    def fit(self) -> None:
        """
        Fit a polynomial model of the specified order to the generated data.
        """
        if self.x is None or self.y is None:
            raise ValueError("Data not generated. Call generate_data() first.")
        self.poly = np.poly1d(np.polyfit(self.x, self.y, self.order))

    def predict(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict target values using the fitted polynomial model.

        Parameters:
            x: Optional custom input array. If None, uses training x.

        Returns:
            Predicted values as a NumPy array.
        """
        if self.poly is None:
            raise RuntimeError("Model must be fit before predicting.")
        return self.poly(x if x is not None else self.x)

    def evaluate_orders(self, orders: List[int]) -> List[float]:
        """
        Compute mean squared error (MSE) for a list of polynomial orders.

        Parameters:
            orders: A list of polynomial degrees.

        Returns:
            List of MSE values corresponding to each order.
        """
        if self.x is None or self.y is None:
            raise ValueError("Data not generated. Call generate_data() first.")

        return [
            mean_squared_error(self.y, np.poly1d(np.polyfit(self.x, self.y, m))(self.x))
            for m in orders
        ]

    def get_fit_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate (x, y) coordinates of the fitted curve for plotting.

        Returns:
            A tuple of (t, poly(t)), where t is a linearly spaced array over the input range.
        """
        if self.poly is None or self.t is None:
            raise RuntimeError("Model must be fit and t-values generated.")
        return self.t, self.poly(self.t)

class PolynomialVisualizer:
    """
    Handles fitting and plotting polynomial models with MSE evaluation.
    """

    def __init__(self, x_range: Tuple[float, float] = (-15, 15)):
        self.x_range = x_range

    def fit_and_plot(
        self,
        x: FloatArray,
        y: FloatArray,
        order: int,
        ax: plt.Axes,
        title_suffix: str = ""
    ) -> float:
        """
        Fit a polynomial and plot it on the given axis.

        Parameters:
            x: Input data.
            y: Target data.
            order: Degree of polynomial.
            ax: Matplotlib axis to plot on.
            title_suffix: Custom title text.

        Returns:
            MSE of the polynomial fit.
        """
        poly = np.poly1d(np.polyfit(x, y, order))
        t = np.linspace(*self.x_range, len(x))
        mse = mean_squared_error(y, poly(x))

        ax.scatter(x, y, c='b', alpha=0.6, edgecolor='k', s=18, label="Noisy Data")
        ax.plot(t, poly(t), 'r-', lw=2, label=f"Fit (m={order})")
        ax.grid(True, ls='--', alpha=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title_suffix}\nMSE={mse:.1f}")
        ax.legend(fontsize=9)

        return mse

    def plot_mse_curve(self, x: FloatArray, y: FloatArray, orders: List[int]) -> None:
        """
        Plot MSE across different polynomial orders.

        Parameters:
            x: Input data.
            y: Target data.
            orders: List of polynomial degrees.
        """
        mse_values = [
            mean_squared_error(y, np.poly1d(np.polyfit(x, y, m))(x)) for m in orders
        ]

        plt.figure(figsize=(9, 6))
        plt.plot(orders, mse_values, "-o", color="royalblue",
                 markerfacecolor='orange', label="MSE")
        plt.grid(True, ls='--', alpha=0.6)
        plt.xlabel("Polynomial Order (m)")
        plt.ylabel("MSE")
        plt.title("MSE vs. Polynomial Order")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_grid_variation(
        self,
        datasets: List[Tuple[FloatArray, FloatArray, str]],
        order: int,
        rows: int,
        cols: int,
        title_prefix: str
    ) -> None:
        """
        Create subplots showing variation (e.g., noise or sample size).

        Parameters:
            datasets: List of (x, y, label) tuples.
            order: Polynomial order to fit.
            rows: Grid rows.
            cols: Grid columns.
            title_prefix: Descriptive prefix for each subplot.
        """
        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axs = axs.flatten()
        for i, (x, y, label) in enumerate(datasets):
            self.fit_and_plot(x, y, order, axs[i], f"{title_prefix}={label}")
        plt.tight_layout()
        plt.show()
