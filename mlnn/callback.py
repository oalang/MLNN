import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.markers import MarkerStyle


class MLNNCallback:
    def __init__(self, print_stats=False, collect_stats=False, animate=False, callback_fun=None):
        self.print_stats = print_stats
        self.collect_stats = collect_stats
        self.animate = animate
        self.callback_fun = callback_fun

        self.optimizer = None
        self.mlnn = None
        self.iter = None
        self.F_prev = None
        self.delta_F = None

        self.stats = None

        self.fig = None
        self.axes = None
        self.artists = None
        self.ani = None

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

        if self.callback_fun is not None:
            self.callback_fun(self.optimizer, self.mlnn, self.iter)

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

        if self.callback_fun is not None:
            self.callback_fun(self.optimizer, self.mlnn, self.iter)

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
        self.stats = []
        self.stats.append(self._stats_dict())

    def _collect_stats_iterate(self):
        self.stats.append(self._stats_dict())

    def _collect_stats_end(self):
        pass

    def _animate_start(self):
        if self.animate == 'projection':
            fig, axes = plt.subplots(1, 1, figsize=(6, 6), squeeze=False)
            axes = axes.ravel()
        elif self.animate == 'all':
            gs_kw = dict(width_ratios=[1, 1], height_ratios=[1, 1])
            fig, axes = plt.subplots(2, 2, figsize=(6, 6), squeeze=False, gridspec_kw=gs_kw)
            axes = axes.ravel()

            axes[1].set_title("Activation Matrix", fontsize=12, family='monospace')
            axes[2].set_title("Weight Matrix", fontsize=12, family='monospace')
            axes[3].set_title("Distance Matrix", fontsize=12, family='monospace')

        axes[0].set_title("2D Projection", fontsize=12, family='monospace')
        axes[0].set_xlim(-0.5, 0.5)
        axes[0].set_ylim(-0.5, 0.5)

        for ax in axes:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_aspect('equal', adjustable='box')

        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.resizable = False

        self.fig = fig
        self.axes = axes
        self.artists = []
        self._draw_new_frame()

    def _animate_iterate(self):
        self._draw_new_frame()

    def _animate_end(self):
        self.ani = ArtistAnimation(fig=self.fig, artists=self.artists, interval=500)
        plt.show()

    def _print_optimize_header(self):
        step = f"{'step':^5s}"
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

        print(" ".join((step, arguments, ls_iterations, alpha, phi, delta_F, F, R, S, L, mean_E, actv_rows, actv_cols, actv_data)))

    def _print_optimize_row(self):
        step = f"{self.iter:5d}"
        arguments = ((f"{self.optimizer.arguments:^4s}" if self.optimizer.arguments is not None else f"{'-':^4s}")
                     if hasattr(self.optimizer, 'arguments') else "")
        ls_iterations = ((f"{self.optimizer.ls_iterations:4d}" if self.optimizer.ls_iterations is not None else f"{'-':^4s}")
                         if hasattr(self.optimizer, 'ls_iterations') else "")
        alpha = ((f"{self.optimizer.alpha:10.3e}" if self.optimizer.alpha is not None else f"{'-':^10s}")
                 if hasattr(self.optimizer, 'alpha') else "" if hasattr(self.optimizer, 'alpha') else "")
        phi = ((f"{self.optimizer.phi:10.3e}" if self.optimizer.phi is not None else f"{'-':^10s}")
               if hasattr(self.optimizer, 'phi') else "")
        delta_F = f"{self.delta_F:10.3e}" if self.delta_F is not None else f"{'-':^10s}"
        F = f"{self.mlnn.F:10.3e}"
        R = f"{self.mlnn.R:10.3e}"
        S = f"{self.mlnn.S:10.3e}"
        L = f"{self.mlnn.L:10.3e}"
        mean_E = f"{np.mean(self.mlnn.E):10.3e}"
        actv_rows = f"{self.mlnn.subset_active_rows.size:9d}"
        actv_cols = f"{self.mlnn.subset_active_cols.size:9d}"
        actv_data = f"{self.mlnn.subset_active_data.size:9d}"

        print(" ".join((step, arguments, ls_iterations, alpha, phi, delta_F, F, R, S, L, mean_E, actv_rows, actv_cols, actv_data)))

    def _stats_dict(self):
        stats_dict = {}
        stats_dict['step'] = self.iter
        if hasattr(self.optimizer, 'arguments'):
            stats_dict['arguments'] = self.optimizer.arguments
        if hasattr(self.optimizer, 'ls_iterations'):
            stats_dict['ls_iterations'] = self.optimizer.ls_iterations
        if hasattr(self.optimizer, 'alpha'):
            stats_dict['alpha'] = self.optimizer.alpha
        if hasattr(self.optimizer, 'phi'):
            stats_dict['phi'] = self.optimizer.phi
        stats_dict['delta_F'] = self.delta_F
        stats_dict['F'] = self.mlnn.F
        stats_dict['R'] = self.mlnn.R
        stats_dict['S'] = self.mlnn.S
        stats_dict['L'] = self.mlnn.L
        stats_dict['mean_E'] = np.mean(self.mlnn.E)
        stats_dict['actv_rows'] = self.mlnn.subset_active_rows.size
        stats_dict['actv_cols'] = self.mlnn.subset_active_cols.size
        stats_dict['actv_data'] = self.mlnn.subset_active_data.size
        stats_dict['time'] = self.optimizer.time

        return stats_dict

    def _draw_new_frame(self):
        frame_artists = self._title_artists(self.fig)
        frame_artists += self._projection_artists(self.axes[0])

        if self.animate == 'all':
            activation_matrix = np.sign(self.mlnn.U)
            frame_artists += self._matrix_artists(self.axes[1], activation_matrix, -1, 1)

            M = self.mlnn.get_transformation_matrix()
            weight_matrix = M.T @ M
            frame_artists += self._matrix_artists(self.axes[2], weight_matrix)

            distance_matrix = self.mlnn.D ** .5
            frame_artists += self._matrix_artists(self.axes[3], distance_matrix)

        self.artists.append(frame_artists)

    def _title_artists(self, fig):
        title = f"i = {self.iter:4d}, f = {self.mlnn.F:12.4f}"
        title_artist = fig.text(0.5, 0.95, title, ha='center', va='bottom', fontsize=12, family='monospace')

        return [title_artist]

    def _projection_artists(self, axis):
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

    def _matrix_artists(self, axis, matrix, vmin=None, vmax=None):
        matrix_artist = axis.imshow(matrix, cmap='gray', vmin=vmin, vmax=vmax)
        return [matrix_artist]
