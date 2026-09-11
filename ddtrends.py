"""
Simple models for trends in density diagnostics

William Henney, whenney@gmail.com, 2026


2026-06-09: Initially extracted from notebook density_diagnostic_trends.ipynb

"""

from typing import Literal, get_args

import numpy as np
from scipy.special import hyp2f1

import pandas as pd


class DimensionlessRatio:
    """
    Normalized line ratio from density-sensitive doublet
    """

    def __init__(self, delta=2.0):
        """
        delta is the sqrt of ratio of the two critical densities
        """
        self.delta = delta

    def __call__(self, ntilde):
        """Normalized line ratio as a function of normalized density"""
        return (1 + ntilde / self.delta) / (1 + ntilde * self.delta)

    def ntilde(self, R):
        "Inverse function giving normalized density from normalized ratio"
        return (1 - R) / (self.delta * R - (1 / self.delta))


class PowerLawPDF:
    """
    Power-law density PDF with upper and lower bounds
    """

    def __init__(self, m: float, nmin: float, nmax: float):
        """
        Parameters: slope `m`, density bounds `nmin`, `nmax`
        """
        self.m = m
        self.nmin = nmin
        self.nmax = nmax
        # Calculate normalization with respect to VEM
        if m == -1:
            self.A = 1 / np.log(self.nmax / self.nmin)
        else:
            self.A = (self.m + 1) / (
                self.nmax ** (self.m + 1) - self.nmin ** (self.m + 1)
            )
        self.label = rf"${m=:.2f}$, $n_\min = {nmin:.2g}$, $n_\max = {nmax:.2g}$"

    def __call__(self, n):
        """Evaluate PDF for density `n` (may be scalar or array)"""
        return np.where((n >= self.nmin) & (n <= self.nmax), self.A * n**self.m, np.nan)

    def F(self, x):
        """Particular case of the Gaussian hypergeometric function that we need for the integrals"""
        # Note that hyp2f1 has a bug for the case of m = -1, where it returns NaNs
        return hyp2f1(1, self.m + 1, self.m + 2, -x)

    def nstats(self):
        """
        Calculate different measures of the central density (variously weighted means and medians)
        """
        rslt = {}
        rslt["Minimum density"] = self.nmin
        rslt["Maximum density"] = self.nmax
        rslt["Mean density"] = self.nmean()
        rslt["RMS density"] = self.nrms()
        rslt["Median density by volume"] = self.ncentile(50, "volume")
        rslt["Median density by emission"] = self.ncentile(50, "emission")
        rslt["05% quantile density by volume"] = self.ncentile(5, "volume")
        rslt["05% quantile density by emission"] = self.ncentile(5, "emission")
        rslt["95% quantile density by volume"] = self.ncentile(95, "volume")
        rslt["95% quantile density by emission"] = self.ncentile(95, "emission")

        return rslt

    def wstats(self):
        """
        Calculate different measures of the distribution width (stddev, clumping, intercentile range)
        """
        rslt = {}
        rslt["Standard deviation of density"] = np.sqrt(
            self.nrms() ** 2 - self.nmean() ** 2
        )
        rslt["Clumping factor"] = self.nrms() ** 2 / self.nmean() ** 2
        rslt["Dex interquantile range by volume 16-84"] = np.log10(
            self.ncentile(84, "volume") / self.ncentile(16, "volume")
        )
        rslt["Dex interquantile range by volume 05-95"] = np.log10(
            self.ncentile(95, "volume") / self.ncentile(5, "volume")
        )
        rslt["Dex interquantile range by emission 16-84"] = np.log10(
            self.ncentile(84, "emission") / self.ncentile(16, "emission")
        )
        rslt["Dex interquantile range by emission 05-95"] = np.log10(
            self.ncentile(95, "emission") / self.ncentile(5, "emission")
        )
        return rslt

    def fstats(self):
        """
        Calculate fraction of emission/volume above different density thresholds
        """
        rslt = {}
        rslt["Emission fraction above 1e4 pcc"] = self.fraction_above(1e4)
        rslt["Emission fraction above 1e5 pcc"] = self.fraction_above(1e5)
        rslt["Emission fraction above 1e6 pcc"] = self.fraction_above(1e6)
        rslt["Volume fraction above 1e4 pcc"] = self.fraction_above(
            1e4, weighting="volume"
        )
        rslt["Volume fraction above 1e5 pcc"] = self.fraction_above(
            1e5, weighting="volume"
        )
        rslt["Volume fraction above 1e6 pcc"] = self.fraction_above(
            1e6, weighting="volume"
        )
        return rslt

    def statistics_table(self):
        """
        Format the statistics as a pandas.Dataframe
        """
        table_rows = {
            "Slope, 𝛽": self.m,
            **self.nstats(),
            **self.wstats(),
            **self.fstats(),
        }.items()
        return pd.DataFrame(
            data=table_rows,
            columns=["Statistic", "Value"],
        ).set_index("Statistic")

    def nrms(self):
        """
        Calculate the RMS density of a bounded power-law distribution

        This is always volume-weighted
        """
        mp1 = self.m + 1
        mm1 = self.m - 1
        n0, n1 = self.nmin, self.nmax
        n2mean = mm1 * (n1**mp1 - n0**mp1)
        n2mean /= mp1 * (n1**mm1 - n0**mm1)
        # Cast to a normal float
        return float(np.sqrt(n2mean))

    def nmean(self):
        """
        Calculate the volumetric mean density of a bounded power-law distribution

        This is always volume-weighted
        """
        m = self.m
        mm1 = self.m - 1
        n0, n1 = self.nmin, self.nmax
        rslt = mm1 * (n1**m - n0**m)
        rslt /= m * (n1**mm1 - n0**mm1)
        return rslt

    def ncentile(self, p, weighting: Literal["emission", "volume"] = "emission"):
        """
        Calculate the `p`-percentile (0-100) of a bounded power-law distribution

        Options for volume or emission (default) weighting
        """
        s = self.m - 2 if weighting == "volume" else self.m
        f = p / 100
        if s == -1.0:
            return self.nmin ** (1.0 - f) * self.nmax**f
        else:
            sp1 = s + 1
            return ((1.0 - f) * self.nmin**sp1 + f * self.nmax**sp1) ** (1 / sp1)

    def fraction_above(self, nthresh, weighting="emission"):
        """
        Fraction of emission or volume above density threshold nthresh.
        """
        if nthresh <= self.nmin:
            return 1.0
        if nthresh >= self.nmax:
            return 0.0

        if weighting == "emission":
            s = self.m
        elif weighting == "volume":
            s = self.m - 2.0
        else:
            raise ValueError(f"Unknown weighting: {weighting!r}")

        if np.isclose(s, -1.0):
            return np.log(self.nmax / nthresh) / np.log(self.nmax / self.nmin)

        sp1 = s + 1.0
        return (self.nmax**sp1 - nthresh**sp1) / (self.nmax**sp1 - self.nmin**sp1)


def stats_from_plaw_params(paramlist: list[tuple[float, float, float]]) -> pd.DataFrame:
    """
    Input `paramlist` is list of (m, log_n0, log_n1) power-law pdf parameters

    Output is dataframe comparing the statistics
    """
    return pd.concat(
        [
            PowerLawPDF(_m, 10**_logn0, 10**_logn1).statistics_table()
            for _m, _logn0, _logn1 in paramlist
        ],
        axis=1,
        ignore_index=True,
    )


class ConcreteRatio:
    """
    A concrete manifestation of the DimensionlessRatio model for a particular line pair

    The difference from DimensionlessRatio is that in addition to the contrast factor `delta`
    we also have a physical density `nM` of max sensitivity and a low-density limit of the line ratio `Rlo`
    (although Rlo is not relevant to the current analysis, so is only useful for plotting).
    Oh, and we also have an optional label. If unspecified, one is created based on the params.
    """

    def __init__(self, nM, delta, Rlo=1.0, label=None):
        self.delta = delta
        self.nM = nM
        self.Rlo = Rlo
        self.Rhi = Rlo / delta**2
        if label is None:
            self.label = f"{nM=:.2e} {delta=:.2f} {Rlo=:.2f}"
        else:
            self.label = label
        self.Rtilde = DimensionlessRatio(delta=delta)

    def __call__(self, n):
        """
        Line ratio as function of physical density `n` (in same units as nM)
        """
        return self.Rlo * self.Rtilde(n / self.nM)

    def n(self, R):
        """
        Inverse function to derive density from line ratio
        """
        return self.nM * self.Rtilde.ntilde(R / self.Rlo)


def apparent_density_powerlaw(R: ConcreteRatio, pdf: PowerLawPDF):
    """
    Find the apparent derived density from a PDF distribution, using a given line ratio diagnostic
    """
    d = R.delta
    nm = R.nM
    n1 = pdf.nmin
    n2 = pdf.nmax
    m = pdf.m
    a2 = n2 ** (m + 1)
    a1 = n1 ** (m + 1)
    F = pdf.F
    # This implements equation PL-4 for the general m case
    numerator = a2 * F(d * n2 / nm) - a1 * F(d * n1 / nm)
    denominator = a2 * F(n2 / nm / d) - a1 * F(n1 / nm / d)
    R_apparent = R.Rlo * numerator / denominator
    assert R.Rhi <= R_apparent <= R.Rlo, "Derived ratio is out of bounds"
    # Invert the ratio to get the "observed" density
    return R.n(R_apparent)


class BimodalPDF:
    """
    Bimodal density PDF
    """

    def __init__(self, frac: float, n0: float, n1: float):
        """
        Parameters: low density `n0`, high density `n1`, fraction of emission from high density `frac`
        """
        self.frac = frac
        self.n0 = n0
        self.n1 = n1
        self.label = rf"$n_0 = {n0:.2g}$, $n_1 = {n1:.2g}$, $f = {frac:.2g}$"

    def nrms(self):
        """
        Calculate the RMS density of a bimodal distribution

        This is always volume-weighted
        """
        ff = self.frac / (1 - self.frac)
        n0_n1 = self.n0 / self.n1
        n2mean = self.n0**2 * (1 + ff) / (1 + n0_n1**2 * ff)
        # Cast to a normal float
        return float(np.sqrt(n2mean))


def apparent_density_bimodal(R: ConcreteRatio, pdf: BimodalPDF):
    """
    Find the apparent derived density from a bimodal PDF distribution, using a given line ratio diagnostic
    """
    d = R.delta
    nm = R.nM
    n0 = pdf.n0
    n1 = pdf.n1
    f = pdf.frac
    numerator = (1 - f) / (nm + d * n0) + f / (nm + d * n1)
    denominator = (1 - f) / (nm + n0 / d) + f / (nm + n1 / d)
    R_apparent = R.Rlo * numerator / denominator
    assert R.Rhi <= R_apparent <= R.Rlo, "Derived ratio is out of bounds"
    # Invert the ratio to get the "observed" density
    return R.n(R_apparent)


def n_obs_eduardo_fit(
    nM: ArrayLike, slope: float = 0.33, intercept: float = 1.34
) -> NDArray[np.floating]:
    """
    Observed linear trend (on log-log scale) of line-ratio-derived density against max-sensitivity density
    of that line ratio (n_M)

    Equation (2) from draft paper 2026-06-03.

    Default slope and intercept are from Eduardo's original fit to Orion Nebula LVM median values (N = 16)
    """
    x = np.log10(nM)
    y = intercept + slope * x
    return 10**y


LineType = Literal["value", "upper", "lower"]
_VALID_LINE_TYPES = get_args(LineType)


def n_obs_improved_fit(
    nM: ArrayLike,
    x0: float = 4.8,
    slope: float = 0.328,
    intercept: float = 2.97,
    e_slope: float = 0.068,
    e_intercept: float = 0.081,
    line_type: LineType = "value",
) -> NDArray[np.floating]:
    """
    Observed linear trend (on log-log scale) of line-ratio-derived density against max-sensitivity density
    of that line ratio (n_M)

    Inspired by Equation (2) from draft paper but with improvements:
    * Non-independent ratios are pruned, reducing N = 16 data points to N = 9
    * Fitted intercept is centered on the data range (log10 n = 4.8) to reduce correlation with fitted slope

    Default slope and intercept are obtained from fit to Orion Nebula LVM median values (N = 9)
    """
    if line_type not in _VALID_LINE_TYPES:
        raise ValueError(
            f"Invalid line_type={line_type!r}; "
            f"expected one of {', '.join(_VALID_LINE_TYPES)}"
        )
    x = np.log10(nM)
    y = intercept + slope * (x - x0)
    if line_type in ("upper", "lower"):
        # Calculate confidence bounds
        sig2 = e_intercept**2 + (e_slope * (x - x0)) ** 2
        # For 9 independent points the 95% confidence band is +/- 2.365 sigma
        dy = 2.365 * np.sqrt(sig2)
        if line_type == "upper":
            y += dy
        else:
            y -= dy
    return 10**y


def n_app_from_nM(
    nM: ArrayLike, pdf: PowerLawPDF | BimodalPDF, delta: ArrayLike | None = None
) -> NDArray[np.floating]:
    """Vectorized version of apparent density"""
    if isinstance(pdf, PowerLawPDF):
        _apparent_density = apparent_density_powerlaw
    elif isinstance(pdf, BimodalPDF):
        _apparent_density = apparent_density_bimodal
    else:
        raise NotImplementedError

    if delta is None:
        # Vary with nM = 1e3 -> 1e7 as delta = 2 -> 200
        delta = 2.0 * np.sqrt(nM / 1e3)
    else:
        # Promote scalar to vector
        delta = delta * np.ones_like(nM)
    return np.array(
        [
            _apparent_density(ConcreteRatio(_nM, delta=_delta), pdf)
            for _nM, _delta in zip(nM, delta)
        ]
    )
