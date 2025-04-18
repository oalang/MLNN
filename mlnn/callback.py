import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.markers import MarkerStyle


class MLNNCallback:
    def __init__(self, print_stats=False, collect_stats=False, animate=False):
        self.print_stats = print_stats
        self.collect_stats = collect_stats
        self.animate = animate

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

        if self.animate:
            self._animate_start()

    def iterate(self, _=None):
        self.iter += 1
        self.delta_F = self.F_prev - self.mlnn.F
        self.F_prev = self.mlnn.F

        if self.print_stats:
            self._print_stats_iterate()

        if self.collect_stats:
            self._collect_stats_iterate()

        if self.animate:
            self._animate_iterate()

    def end(self):
        if self.print_stats:
            self._print_stats_end()

        if self.collect_stats:
            self._collect_stats_end()

        if self.animate:
            self._animate_end()

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

    def _animate_start(self):
        gs_kw = dict(width_ratios=[1, 1], height_ratios=[1, 1])
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4))  = plt.subplots(2, 2, figsize=(6, 6), gridspec_kw=gs_kw)
        self.ax1.set_title("2D Projection", fontsize=12, family='monospace')
        self.ax1.xaxis.set_visible(False)
        self.ax1.yaxis.set_visible(False)
        self.ax1.set_aspect('equal', adjustable='box')
        self.ax1.set_xlim(-0.5, 0.5)
        self.ax1.set_ylim(-0.5, 0.5)
        self.ax2.set_title("Activation", fontsize=12, family='monospace')
        self.ax2.xaxis.set_visible(False)
        self.ax2.yaxis.set_visible(False)
        self.ax2.set_aspect('equal', adjustable='box')
        self.ax3.set_title("Weight Matrix", fontsize=12, family='monospace')
        self.ax3.xaxis.set_visible(False)
        self.ax3.yaxis.set_visible(False)
        self.ax3.set_aspect('equal', adjustable='box')
        self.ax4.set_title("Distance Matrix", fontsize=12, family='monospace')
        self.ax4.xaxis.set_visible(False)
        self.ax4.yaxis.set_visible(False)
        self.ax4.set_aspect('equal', adjustable='box')
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.resizable = False
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.8)

        self.artists = []

        title_artists = self._figure_title_artists(self.fig)
        scatter_plot_artists = self._scatter_plot_artists(self.ax1)
        activation_image_artists = self._activation_image_artists(self.ax2)
        weight_matrix_artists = self._weight_matrix_artists(self.ax3)
        distance_matrix_artists = self._distance_matrix_artists(self.ax4)
        self.artists.append(title_artists + scatter_plot_artists + activation_image_artists + weight_matrix_artists + distance_matrix_artists)

    def _animate_iterate(self):
        title_artists = self._figure_title_artists(self.fig)
        scatter_plot_artists = self._scatter_plot_artists(self.ax1)
        activation_image_artists = self._activation_image_artists(self.ax2)
        weight_matrix_artists = self._weight_matrix_artists(self.ax3)
        distance_matrix_artists = self._distance_matrix_artists(self.ax4)
        self.artists.append(title_artists + scatter_plot_artists + activation_image_artists + weight_matrix_artists + distance_matrix_artists)

    def _animate_end(self):
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

    def _figure_title_artists(self, fig):
        title = f"i = {self.iter:3d}, f = {self.mlnn.F:10.3f}"
        title_artist = fig.text(0.5, 0.90, title, ha='center', va='bottom', fontsize=12, family='monospace')

        return [title_artist]

    def _scatter_plot_artists(self, axis):
        M = self.mlnn.get_transformation_matrix(n_components=2)
        X = self.mlnn.B @ M.T
        Y = self.mlnn.Y
        A = np.full(X.shape[0], False)
        A[self.mlnn.subset_active_rows] = True

        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        x_range = x_max - x_min
        y_range = y_max - y_min
        scale = 0.9 / max(x_range, y_range)
        X = (X - np.array([x_center, y_center])) * scale

        Y_unique = np.sort(np.unique(Y))
        colors = plt.cm.tab10(np.linspace(0, 1, len(Y_unique)))
        color_map = dict(zip(Y_unique, colors))
        C = np.array([color_map[y] for y in Y])

        inactive_artist = axis.scatter(X[~A, 0], X[~A, 1], c=C[~A], marker=MarkerStyle('o', fillstyle='none'))
        active_artist = axis.scatter(X[A, 0], X[A, 1], c=C[A], marker=MarkerStyle('o', fillstyle='full'))

        return [inactive_artist, active_artist]

    def _activation_image_artists(self, axis):
        image_artist = axis.imshow(np.sign(self.mlnn.U), cmap='gray', vmin=-1, vmax=1)
        return [image_artist]

    def _weight_matrix_artists(self, axis):
        M = self.mlnn.get_transformation_matrix()
        W = M.T @ M
        image_artist = axis.imshow(W, cmap='gray', vmin=W.min(), vmax=W.max())
        return [image_artist]

    def _distance_matrix_artists(self, axis):
        D = self.mlnn.D ** .5
        image_artist = axis.imshow(D, cmap='gray', vmin=D.min(), vmax=D.max())
        return [image_artist]
