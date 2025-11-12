"""
Calculate the density PDF of a single photoevaporation flow
from a globule, filament, or proplyd

Will Henney 2025-11-11
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Literal


class Flow:
    """
    A single steady-state photoevaporation flow
    """

    Umax = 4.0  # Maximum Mach number for grid
    NR = 1000

    def __init__(self, ndim: int = 3):
        """
        A single photoevaporation flow that is:
        1. steady-state
        2. isothermal
        3. purely radial
        4. fully-ionized
        5. from a D-critical front (M = 1 at R = 1)
        6. non-magnetized

        Parameters
        ----------
            ndim: The number of spatial dimensions - either 2 (cylindrical) or 3 (spherical, default)
        """
        # Set the divergence index alpha, such that rho u (r ** alpha) = constant
        if ndim == 3:
            # Spherical globule
            self.alpha = 2.0
        elif ndim == 2:
            # Cylindrical filament
            self.alpha = 1.0
        else:
            raise ValueError("Number of dimensions must be 2 or 3")

        # Initialise grids of Mach numbers, radii, densities
        # We will consider these points as the cell edges
        self.U_edges = np.linspace(1.0, self.Umax, self.NR, endpoint=True)
        self.R_edges = self.dyson_R_U(self.U_edges)
        self.density_edges = self.density_U(self.U_edges)

    def dyson_R_U(self, U: ArrayLike) -> NDArray[np.floating]:
        """Dimensionless radius as function of Mach number

        Dyson solution R(U) for isothermal evaporation flow from D-critical front.
        This derives from the isothermal Bernoulli equation and continuity in ndim dimensions
        """
        # Find radius in terms of Mach number
        return U ** (-1 / self.alpha) * np.exp((U**2 - 1) / (2 * self.alpha))

    def density_U(self, U: ArrayLike) -> NDArray[np.floating]:
        """Density as function of Mach number"""
        return 1 / U / self.dyson_R_U(U) ** self.alpha

    def pressure_U(self, U: ArrayLike) -> NDArray[np.floating]:
        """Dynamic pressure (thermal plus ram) as function of Mach number"""
        return (1 + U**2) * self.density_U(U)

    def radial_density_histogram(
        self,
        radii: np.ndarray[np.floating],
        densities: np.ndarray[np.floating],
        nbins: int = 100,
        min_density: float = 0.0,
        density_scale: Literal["linear", "log"] = "linear",
    ) -> NDArray[np.floating]:
        """
        PDF histogram of density variation with radius, weighted by density-squared times volume element

        Either a linear or logarithmic density scale can be used (density_scale = 'linear' or 'log')
        """
        radii_centers = 0.5 * (radii[:-1] + radii[1:])
        volume_elements = radii_centers**self.alpha * np.diff(radii)
        density_centers = 0.5 * (densities[:-1] + densities[1:])
        weights = density_centers**2 * volume_elements
        if density_scale == "linear":
            H, edges = np.histogram(
                density_centers,
                bins=nbins,
                range=(min_density, 1.0),
                weights=weights,
                density=True,
            )
        elif density_scale == "log":
            H, edges = np.histogram(
                np.log10(density_centers),
                bins=nbins,
                range=(np.log10(min_density), 0.0),
                weights=weights,
                density=True,
            )
        else:
            raise ValueError("density_scale must be 'linear' or 'log'")
        return H, edges
