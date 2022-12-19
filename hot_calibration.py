import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure


class hot_calibration:
    def __init__(self, temperature=296, unit='pN/um', mag=60, pixel=5.86, magEx=False, potential_bin_count=50):
        self.n = 0
        self.temperature = temperature
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

    def gauss_distribution(self, x, kx, xeq=0):
        rho = np.sqrt(kx / 2 / np.pi / self.kB / self.temperature) * np.exp(
            -kx / 2 / self.kB / self.temperature * (x - xeq) ** 2)
        return rho

    def potential_analysis(self, xs, multi=False, showplot=False):
        xs = xs * self.img_pixel_size
        xmean = np.mean(xs)
        xc = xs
        xmin = np.min(xc)
        xmax = np.max(xc)
        xc_hist, xc_bins = np.histogram(xc, bins=self.bin_count, density=True, range=(xmin, xmax))
        binw = np.diff(xc_bins)[0]
        x_coord = np.arange(xmin + binw / 2, xmax, binw)
        # print(xc_hist.shape,x_coord.shape)
        kx, xeq = curve_fit(self.gauss_distribution, xdata=x_coord, ydata=xc_hist, p0=(self.equipartition(xs),xmean))[0]
        print(f'k={kx}, eq={xeq}')
        if showplot:
            fig = plt.figure(figsize=(12, 9))
            ax1 = fig.add_subplot(111)  # type:axes.Axes
            ax1.plot(x_coord, xc_hist, 'b.')
            ax1.plot(x_coord, self.gauss_distribution(x_coord, kx, xeq), 'r')
            plt.show()
        return kx
