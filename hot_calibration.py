import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
from matplotlib import gridspec


class hot_calibration:
    def __init__(self, temperature=296, unit='pN/um', mag=60, pixel=5.86, magEx=False, potential_bin_count=50):
        self.n = 0
        self.temperature = temperature
        self.unitstr = unit
        if unit == 'pN/um':
            # k_Boltzmann = 1.380649 * 10^-5 pN um K^-1
            self.kB = 1.380649 * 10 ** -5
        else:
            # k_Boltzmann = 1.380649 * 10^-23 J K^-1
            self.kB = 1.380649 * 10 ** -23
        self.img_pixel_size = pixel / mag
        self.img_pixel_size = self.img_pixel_size / 1.5 if magEx else self.img_pixel_size
        self.bin_count = potential_bin_count

    def equipartition(self, xs, multi=False, showplot=False):
        # xmean = np.mean(xs)
        # xc = xs - xmean
        xs = xs * self.img_pixel_size
        if multi:
            xvar = np.var(xs, axis=1)
            ks = self.kB * self.temperature / xvar
            kappa = np.mean(ks)
        else:
            xvar = np.var(xs)
            # print(xvar)
            kappa = self.kB * self.temperature / xvar
        return kappa

    def gauss_distribution(self, x, kx):
        rho = np.sqrt(kx / 2 / np.pi / self.kB / self.temperature) * np.exp(
            -kx / 2 / self.kB / self.temperature * (x) ** 2)
        return rho

    def potential_linear_funct(self, x, a, b):
        y = -a * x ** 2 + b
        return y

    def potential_analysis(self, xs, multi=False, showplot=False):
        xs = xs * self.img_pixel_size
        xmean = np.mean(xs)
        xc = xs - xmean
        xmin = np.min(xc)
        xmax = np.max(xc)
        xc_hist, xc_bins = np.histogram(xc, bins=self.bin_count, density=True, range=(xmin, xmax))

        binw = np.diff(xc_bins)[0]
        x_coord = np.arange(xmin + binw / 2, xmax, binw)

        kx, = curve_fit(self.gauss_distribution, xdata=x_coord, ydata=xc_hist, p0=(self.equipartition(xs)),
                        bounds=(0, np.inf))[0]
        print(f'k={kx}, eq={xmean}')
        if showplot:
            fig = plt.figure(figsize=(12, 9))  # type:figure.Figure
            ax1 = fig.add_subplot(111)  # type:axes.Axes
            ax1.hist(xc, bins=self.bin_count, density=True, color='C0')
            ax1.plot(x_coord, self.gauss_distribution(x_coord, kx), 'r')
            plt.show()
        return kx

    def potential_analysis_linear(self, xs, multi=False, showplot=False):
        xs = xs * self.img_pixel_size
        xmean = np.mean(xs)
        xc = xs - xmean
        xmin = np.min(xc)
        xmax = np.max(xc)
        xc_hist, xc_bins = np.histogram(xc, bins=self.bin_count, density=True, range=(xmin, xmax))
        nzero = np.where(xc_hist != 0)
        binw = np.diff(xc_bins)[0]
        x_coord = np.arange(xmin + binw / 2, xmax, binw)
        x_coord = x_coord[nzero]
        ln_rho = np.log(xc_hist[nzero])
        a, b = curve_fit(self.potential_linear_funct, xdata=x_coord, ydata=ln_rho, bounds=(0, np.inf))[0]
        k1 = 2 * a * self.kB * self.temperature
        k2 = 2 * np.pi * self.kB * self.temperature * np.exp(2 * b)
        print(f'k from a: {k1}; k from b: {k2}')
        if showplot:
            fig = plt.figure(figsize=(12, 9))  # type:figure.Figure
            ax1 = fig.add_subplot(111)  # type:axes.Axes
            ax1.hist(xc, bins=self.bin_count, density=True, color='C0')
            ax1.plot(x_coord, self.gauss_distribution(x_coord, k1), 'r')
            plt.show()
        return k1

    def eq_pa(self, xs, t, showplot=False):
        xs = xs * self.img_pixel_size
        xmean = np.mean(xs)
        xc = xs - xmean
        xvar = np.var(xs)
        keq = self.kB * self.temperature / xvar

        xmin = np.min(xc)
        xmax = np.max(xc)
        xc_hist, xc_bins = np.histogram(xc, bins=self.bin_count, density=True, range=(xmin, xmax))

        binw = np.diff(xc_bins)[0]
        x_coord = np.arange(xmin + binw / 2, xmax, binw)
        kp, = curve_fit(self.gauss_distribution, xdata=x_coord, ydata=xc_hist, p0=(keq),
                        bounds=(0, np.inf))[0]

        nzero = np.where(xc_hist != 0)
        ln_rho = np.log(xc_hist[nzero])
        a, b = curve_fit(self.potential_linear_funct, xdata=x_coord[nzero], ydata=ln_rho, bounds=(0, np.inf))[0]
        kpa1 = 2 * a * self.kB * self.temperature
        kpa2 = 2 * np.pi * self.kB * self.temperature * np.exp(2 * b)
        if showplot:
            fig = plt.figure(figsize=(12, 9))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            ax = plt.subplot(gs[0])  # type:axes.Axes
            ax.hist(xc, bins=self.bin_count, density=True, color='C0', label='Positional distribution')
            ax.plot(x_coord, self.gauss_distribution(x_coord, keq), 'r', label='Equipartition')
            ax.plot(x_coord, self.gauss_distribution(x_coord, kp), 'c', label='Potential Analysis')
            ax.plot(x_coord, self.gauss_distribution(x_coord, kpa1), 'm', label='Potential Analysis alter. (a)')
            ax.plot(x_coord, self.gauss_distribution(x_coord, kpa2), 'orchid', label='Potential Analysis alter. (b)')
            ax.legend(loc='upper right')
            ax.set_xlabel('Displacement [um]')
            ax.set_ylabel('Normalized Distribution')
            ax.set_title('Particle Position and Calibration Curves')
            ax = plt.subplot(gs[1])
            ax.plot(t, xs, 'C0')
            ax.set_xlabel('Time [ms]')
            ax.set_ylabel('Centroid [um]')
            # ax.set_title('Particle Trajectory')
            plt.show()
        print(f'k_eq={keq:.5} {self.unitstr}, k_p={kp:.5} {self.unitstr}, k_pa={kpa1:.5} / {kpa2:.5} {self.unitstr}.')
        return keq, kp, kpa1, kpa2
