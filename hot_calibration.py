import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
from matplotlib import gridspec
import uncertainties as u
import uncertainties.umath as umath


class hot_calibration:
    def __init__(self, temperature=296, unit='pN/um', mag=60, pixel=5.86, magEx=False, potential_bin_count=50):
        self.n = 0
        self.temperature = temperature
        self.unitstr = unit
        if unit == 'pN/um':
            # k_Boltzmann = 1.380649 * 10^-5 pN um K^-1
            self.kB = 1.380649 * 10 ** -5
            self.len_unit_str = 'um'
        else:
            # k_Boltzmann = 1.380649 * 10^-23 J K^-1
            self.kB = 1.380649 * 10 ** -23
            self.len_unit_str = 'm'
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

    def gauss_distribution_offset(self, x, kx, xeq):
        rho = np.sqrt(kx / 2 / np.pi / self.kB / self.temperature) * np.exp(
            -kx / 2 / self.kB / self.temperature * (x - xeq) ** 2)
        return rho

    def potential_linear_funct(self, x, a, b):
        y = -a * x ** 2 + b
        return y

    def potential_linear_funct_offset(self, x, a, b, xeq):
        y = -a * (x - xeq) ** 2 + b
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
        if abs(xmin) > abs(xmax):
            xmin = -xmax
            # print('changed min:', xmin, xmax)
        else:
            xmax = - xmin
            # print('changed max', xmin, xmax)
        xc_hist, xc_bins = np.histogram(xc, bins=self.bin_count, density=True, range=(xmin, xmax))

        binw = np.diff(xc_bins)[0]
        x_coord = np.arange(xmin + binw / 2, xmax, binw)
        try:
            popt, pcov, infodict, mesg, ier = curve_fit(self.gauss_distribution_offset, xdata=x_coord, ydata=xc_hist,
                                                        p0=(keq, 0),
                                                        bounds=(0, np.inf), full_output=True)
            kp = popt[0]
            xeq = popt[1]
            perr = np.sqrt(np.diag(pcov))
            kp_err = perr[0]
            xeq_err = perr[1]
            ss_err = (infodict['fvec'] ** 2).sum()
            ss_tot = ((xc_hist - xc_hist.mean()) ** 2).sum()
            p_rsqr = 1 - (ss_err / ss_tot)
            fit_p_success = True
        except:
            kp = 0
            xeq = 0
            kp_err = 0
            xeq_err = 0
            p_rsqr = 0
            print('Potential Analysis fit to Gaussian Distribution: Failed.')
            fit_p_success = False

        nzero = np.where(xc_hist != 0)
        ln_rho = np.log(xc_hist[nzero])
        try:
            popt, pcov, infodict, mesg, ier = curve_fit(self.potential_linear_funct_offset, xdata=x_coord[nzero],
                                                        ydata=ln_rho,
                                                        bounds=(0, np.inf), full_output=True)
            a = popt[0]
            b = popt[1]
            xeqpa = popt[2]
            perr = np.sqrt(np.diag(pcov))
            a_err = perr[0]
            b_err = perr[1]
            xeqpa_err = perr[2]
            a_u = u.ufloat(a, a_err)
            b_u = u.ufloat(b, b_err)
            kpa1_u = 2 * a_u * self.kB * self.temperature
            kpa2_u = 2 * np.pi * self.kB * self.temperature * umath.exp(2 * b_u)
            kpa1 = kpa1_u.nominal_value
            kpa2 = kpa2_u.nominal_value
            kpa1_err = kpa1_u.std_dev
            kpa2_err = kpa2_u.std_dev
            ss_err = (infodict['fvec'] ** 2).sum()
            ss_tot = ((xc_hist[nzero] - xc_hist[nzero].mean()) ** 2).sum()
            pa_rsqr = 1 - (ss_err / ss_tot)
            fit_pa_success = True
        except:
            fit_pa_success = False
            kpa1 = 0
            kpa2 = 0
            kpa2_err = 0
            kpa1_err = 0
            xeqpa = 0
            xeqpa_err = 0
            pa_rsqr = 0
            print('Potential Analysis fit to polynomial Gaussian Distribution: Failed.')

        if showplot:
            fig = plt.figure(figsize=(12, 9))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            ax = plt.subplot(gs[0])  # type:axes.Axes
            ax.hist(xc, bins=self.bin_count, density=True, color='C0', range=(xmin, xmax),
                    label='Positional distribution')
            ax.plot(x_coord, self.gauss_distribution(x_coord, keq), 'r', label='Equipartition')
            if fit_p_success:
                ax.plot(x_coord, self.gauss_distribution_offset(x_coord, kp, xeq), 'c', label='Potential Analysis')
            if fit_pa_success:
                ax.plot(x_coord, self.gauss_distribution_offset(x_coord, kpa1, xeqpa), 'm',
                        label='Potential Analysis alter. (a)')
                ax.plot(x_coord, self.gauss_distribution_offset(x_coord, kpa2, xeqpa), 'orchid',
                        label='Potential Analysis alter. (b)')
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
        print(f'k_eq = {keq:.5} {self.unitstr};')
        print(f'k_p = {kp:.5} ± {kp_err:.5} {self.unitstr}, x_eq = {xeq:.5} ± {xeq_err:.5} {self.len_unit_str}, ',
              f'fit R^2={p_rsqr:.5};')
        print(f'k_pa = {kpa1:.5} ± {kpa1_err:.5} / {kpa2:.5} ± {kpa2_err:.5} {self.unitstr}',
              f'xeq = {xeqpa:.5} ± {xeqpa_err:.5} {self.len_unit_str}, fit R^2={pa_rsqr:.5}.')
        return keq, kp, kp_err, xeq, xeq_err, p_rsqr, kpa1, kpa1_err, kpa2, kpa2_err, xeqpa, xeqpa_err, pa_rsqr
