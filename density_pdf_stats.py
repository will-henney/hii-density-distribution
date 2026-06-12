# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Density PDF statistics
#
# Investigating descriptive statistics of density distributions, in particular the bounded power law. 
#
# This notebook will serve as a testbed for code that will eventually be incorporated into the `ddtrends` module for use in the other notebooks. 

# +
from typing import Literal, get_args

import numpy as np
import pandas as pd
from astropy.table import Table


# -

# ## The PowerLawPDF class and its methods

# We have to keep the whole class definition in a single cell

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
            self.nrms() ** 2 - self.nmean() **2
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

    def statistics_table(self):
        """
        Format the statistics as a pandas.Dataframe
        """
        table_rows = {"Slope, 𝛽": self.m, **self.nstats(), **self.wstats()}.items()
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


# ## Tests of the characteristic densities and widths

# First try the best-fit model from MCMC (median parameters)

pdf = PowerLawPDF(-1.3, 10**1.5, 10**6.3)
# pdf = PowerLawPDF(-1.3, 20, 3e6)


pdf.statistics_table()

# So for this example, the mean is about twice the minimum density and the rms is about 3 times the minimum density. The median by volume is in between the two, but the median by emission is much larger (10 times the minimum). The 95% quantile by emission is enormous: 3000 times the minimum.
#
# The standard deviation is of the same order as the mean (which makes it not a good measure of the distribution width). 

#

# ### Experiment with different ways of displaying the stats

# Raw dict

pdf.nstats()

# Pandas dataframe

# +
pd.set_option('display.float_format', '{:.4g}'.format)

(
    pd.DataFrame(
        data={"Slope, 𝛽": pdf.m, **pdf.nstats(), **pdf.wstats()}.items(), 
        columns=["Statistic", "Value"],
    )
    .set_index("Statistic")
    # .style.format(
    #     {"Value": "{:.4g}"},
    # )
)
# -

# Astropy Table

tab = Table(rows=pdf.nstats().items(), names=["Statistic", "Value"])
tab["Value"].format = "{:.4g}"
tab

pd.concat(
    [
        PowerLawPDF(-1.4, 10**1.7, 10**7.4).statistics_table(),
        PowerLawPDF(-1.0001, 10**0.0, 10**4.7).statistics_table(),
    ],
    axis=1,
)

PowerLawPDF(-0.99, 10**1.5, 10**6.3).nmean()


# ## Comparison of different parameter sets

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


# ### Varying the slope

stats_from_plaw_params(
    [
        (-1.6, 1.7, 7.0),
        (-1.4, 1.7, 7.0),
        (-1.2, 1.7, 7.0),
        (-1.0001, 1.7, 7.0),
        (-0.8, 1.7, 7.0),
        (-0.6, 1.7, 7.0),
    ]
)

# ### Varying the lower bound

stats_from_plaw_params(
    [
        (-1.4, 0.7, 7.0),
        (-1.4, 1.2, 7.0),
        (-1.4, 1.7, 7.0),
        (-1.4, 2.2, 7.0),
        (-1.4, 2.7, 7.0),
    ]
)

# The mean and rms are greatly affected by the minimum density

# ### Varying the upper bound

stats_from_plaw_params(
    [
        (-1.4, 1.7, 5.0),
        (-1.4, 1.7, 6.0),
        (-1.4, 1.7, 7.0),
        (-1.4, 1.7, 8.0),
        (-1.4, 1.7, 9.0),
    ]
)

# Varying the maximum density makes very little difference to the majority of the measures.
#
# The dex interquantile ranges go down a bit for max densies below 1e6
#
# And the 95% quantile does increase slowly with n_1

# +
# pd.concat?
# -


