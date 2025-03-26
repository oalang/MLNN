
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.markers import MarkerStyle


class MLNNCallback:
    def __init__(self, print_stats=False, collect_stats=False, show_figures=False):
        self.print_stats = print_stats
        self.collect_stats = collect_stats
        self.show_figures = show_figures

        self.optimizer = None
        self.mlnn = None
        self.iter = None
        self.F_prev = None
        self.delta_F = None

    def start(self, optimizer):
        self.optimizer = optimizer
        self.mlnn = optimizer.mlnn
        self.iter = 0
        self.F_prev = self.mlnn.F

        if self.print_stats:
            self._print_stats_start()

        if self.collect_stats:
            self._collect_stats_start()

        if self.show_figures:
            self._show_figures_start()

    def iterate(self, _=None):
        self.iter += 1
        self.delta_F = self.F_prev - self.mlnn.F
        self.F_prev = self.mlnn.F

        if self.print_stats:
            self._print_stats_iterate()

        if self.collect_stats:
            self._collect_stats_iterate()

        if self.show_figures:
            self._show_figures_iterate()

    def end(self):
        if self.print_stats:
            self._print_stats_end()

        if self.collect_stats:
            self._collect_stats_end()

        if self.show_figures:
            self._show_figures_end()

    def _print_stats_start(self):
        self._print_optimize_header()
        self._print_optimize_row()

    def _print_stats_iterate(self):
        self._print_optimize_row()

    def _print_stats_end(self):
        pass

    def _collect_stats_start(self):
        pass

    def _collect_stats_iterate(self):
        pass

    def _collect_stats_end(self):
        pass

    def _show_figures_start(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.artists = []

        M = self.mlnn.get_transformation_matrix(n_components=2)
        self.M_prev = M
        X = self.mlnn.B
        X = X @ M.T
        active = np.full(X.shape[0], False)
        active[self.mlnn.subset_active_rows] = True
        artist1 = self.ax1.scatter(X[~active, 0], X[~active, 1], c=np.ones(np.sum(~active)), marker=MarkerStyle('o', fillstyle='none'))
        artist2 = self.ax1.scatter(X[active, 0], X[active, 1], c=np.ones(np.sum(active)), marker=MarkerStyle('o', fillstyle='full'))
        artist3 = self.ax2.imshow(np.sign(self.mlnn.U),
                                cmap='gray', vmin=-1, vmax=1)
        self.artists.append((artist1, artist2, artist3))

    def _show_figures_iterate(self):
        M = self.mlnn.get_transformation_matrix(n_components=2)
        S = np.sign(np.sum(M * self.M_prev, axis=1, keepdims=True))
        #M *= np.where(S == 0, 1, S)
        self.M_prev = M
        X = self.mlnn.B
        X = X @ M.T
        active = np.full(X.shape[0], False)
        active[self.mlnn.subset_active_rows] = True
        artist1 = self.ax1.scatter(X[~active, 0], X[~active, 1], c=np.ones(np.sum(~active)), marker=MarkerStyle('o', fillstyle='none'))
        artist2 = self.ax1.scatter(X[active, 0], X[active, 1], c=np.ones(np.sum(active)), marker=MarkerStyle('o', fillstyle='full'))
        artist3 = self.ax2.imshow(np.sign(self.mlnn.U),
                                cmap='gray', vmin=-1, vmax=1)
        self.artists.append((artist1, artist2, artist3))

    def _show_figures_end(self):
        self.ani = ArtistAnimation(fig=self.fig, artists=self.artists, interval=500)
        plt.show()

    def _print_optimize_header(self):
        steps = f"{'step':^5s}"
        arguments = f"{'args':^4s}" if hasattr(self.optimizer, 'arguments') else ""
        ls_iterations = f"{'iter':^4s}" if hasattr(self.optimizer, 'ls_iterations') else ""
        alpha = f"{'alpha':^10s}" if hasattr(self.optimizer, 'alpha') else ""
        phi = f"{'phi':^10s}" if hasattr(self.optimizer, 'phi') else ""
        delta_F = f"{'delta_F':^10s}"
        F = f"{'F':^10s}"
        R = f"{'R':^10s}"
        S = f"{'S':^10s}"
        L = f"{'L':^10s}"
        mean_E = f"{'mean_E':^10s}"
        actv_rows = f"{'actv_rows':^9s}"
        actv_cols = f"{'actv_cols':^9s}"
        actv_data = f"{'actv_data':^9s}"

        print(" ".join((steps, arguments, ls_iterations, alpha, phi, delta_F, F, R, S, L, mean_E, actv_rows, actv_cols, actv_data)))

    def _print_optimize_row(self):
        steps = f"{self.iter:5d}" if self.iter is not None else f"{'-':^5s}"
        arguments = ((f"{self.optimizer.arguments:^4s}" if self.optimizer.arguments is not None else f"{'-':^4s}")
                     if hasattr(self.optimizer, 'arguments') else "")
        ls_iterations = ((f"{self.optimizer.ls_iterations:4d}" if self.optimizer.ls_iterations is not None else f"{'-':^4s}")
                         if hasattr(self.optimizer, 'ls_iterations') else "")
        alpha = ((f"{self.optimizer.alpha:10.3e}" if self.optimizer.alpha is not None else f"{'-':^10s}")
                 if hasattr(self.optimizer, 'alpha') else "" if hasattr(self.optimizer, 'alpha') else "")
        phi = ((f"{self.optimizer.phi:10.3e}" if self.optimizer.phi is not None else f"{'-':^10s}")
               if hasattr(self.optimizer, 'phi') else "")
        delta_F = f"{self.delta_F:10.3e}" if self.delta_F is not None else f"{'-':^10s}"
        F = f"{self.mlnn.F:10.3e}" if self.mlnn.F is not None else f"{'-':^10s}"
        R = f"{self.mlnn.R:10.3e}" if self.mlnn.R is not None else f"{'-':^10s}"
        S = f"{self.mlnn.S:10.3e}" if self.mlnn.S is not None else f"{'-':^10s}"
        L = f"{self.mlnn.L:10.3e}" if self.mlnn.L is not None else f"{'-':^10s}"
        mean_E = f"{np.mean(self.mlnn.E):10.3e}" if self.mlnn.E is not None else f"{'-':^10s}"
        actv_rows = (f"{self.mlnn.subset_active_rows.size:9d}"
                     if self.mlnn.subset_active_rows.size is not None else f"{'-':^9s}")
        actv_cols = (f"{self.mlnn.subset_active_cols.size:9d}"
                     if self.mlnn.subset_active_cols.size is not None else f"{'-':^9s}")
        actv_data = (f"{self.mlnn.subset_active_data.size:9d}"
                     if self.mlnn.subset_active_data.size is not None else f"{'-':^9s}")

        print(" ".join((steps, arguments, ls_iterations, alpha, phi, delta_F, F, R, S, L, mean_E, actv_rows, actv_cols, actv_data)))
