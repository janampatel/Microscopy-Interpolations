"""Result viewer for the GUI: playback, A|B compare, overview and, for validation
runs, per-frame PSNR/SSIM.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QCheckBox, QComboBox, QTabWidget, QGridLayout, QScrollArea, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PySide6.QtGui import QImage, QPixmap

try:
    import pyqtgraph as pg
    _HAVE_PG = True
except Exception:
    _HAVE_PG = False


ACCENT = "#e67e22"      # interpolated highlight
OK_GREEN = "#27ae60"    # original / ground truth
BG = "#1e1e1e"


# --------------------------------------------------------------------------- utils
def to_display(img: np.ndarray) -> np.ndarray:
    """Normalize any 2D array to float32 in [0, 1] for display (per-frame)."""
    a = np.asarray(img).astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        return (a - lo) / (hi - lo)
    return np.zeros_like(a)


def to_qpixmap(img: np.ndarray, max_side: int | None = None) -> QPixmap:
    """8-bit grayscale QPixmap from a 2D array (optionally down-scaled)."""
    disp = (to_display(img) * 255).astype(np.uint8)
    disp = np.ascontiguousarray(disp)
    h, w = disp.shape
    qim = QImage(disp.data, w, h, w, QImage.Format_Grayscale8).copy()
    pm = QPixmap.fromImage(qim)
    if max_side and (w > max_side or h > max_side):
        pm = pm.scaled(max_side, max_side, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pm


# ------------------------------------------------------------------- image pane
class ImagePane(QWidget):
    """A single zoom/pan-able grayscale image view.

    Uses pyqtgraph when available (interactive zoom/pan, levels); otherwise
    falls back to a fit-to-window QLabel. `diff=True` applies a heat colormap.
    """

    def __init__(self, title: str = "", diff: bool = False):
        super().__init__()
        self._diff = diff
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.header = QLabel(title)
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet("color:#ddd; font-size:12px; font-weight:600; padding:2px;")
        layout.addWidget(self.header)

        if _HAVE_PG:
            self.glw = pg.GraphicsLayoutWidget()
            self.glw.setBackground(BG)
            self.vb = self.glw.addViewBox()
            self.vb.setAspectLocked(True)
            self.vb.invertY(True)
            self.vb.setMenuEnabled(False)
            self.item = pg.ImageItem(axisOrder="row-major")
            self.vb.addItem(self.item)
            if diff:
                try:
                    self.item.setLookupTable(pg.colormap.get("inferno").getLookupTable(0.0, 1.0, 256))
                except Exception:
                    pass
            layout.addWidget(self.glw, 1)
        else:
            self.label = QLabel("(install pyqtgraph for interactive view)")
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setStyleSheet(f"background:{BG}; border:1px solid #333;")
            layout.addWidget(self.label, 1)

    def set_image(self, img):
        if img is None:
            return
        disp = to_display(img)
        if _HAVE_PG:
            self.item.setImage(disp, autoLevels=False, levels=(0.0, 1.0))
        else:
            self.label.setPixmap(to_qpixmap(img, max_side=700))

    def link_to(self, other: "ImagePane"):
        """Synchronize zoom/pan of this pane to another (pyqtgraph only)."""
        if _HAVE_PG:
            self.vb.setXLink(other.vb)
            self.vb.setYLink(other.vb)


def _badge(text: str, color: str) -> QLabel:
    lab = QLabel(text)
    lab.setAlignment(Qt.AlignCenter)
    lab.setStyleSheet(
        f"color:white; background:{color}; border-radius:9px; padding:2px 10px; font-weight:700; font-size:11px;")
    return lab


# ----------------------------------------------------------------- main dialog
class InterpolationVisualizer(QDialog):
    def __init__(self, output, roles, gt=None, metrics=None, backend="", parent=None):
        """
        output  : (M, H, W) hybrid result stack
        roles   : list length M of {'role': 'original'|'interpolated', ...}
        gt      : optional {output_index: gt_frame} (validation)
        metrics : optional {output_index: (psnr, ssim)} (validation)
        """
        super().__init__(parent)
        self.output = output
        self.roles = roles
        self.gt = gt or {}
        self.metrics = metrics or {}
        self.backend = backend
        self.interp_indices = [i for i, r in enumerate(roles) if r.get("role") == "interpolated"]

        mode = "Validation" if self.gt else "Result"
        self.setWindowTitle(f"microinterp - {mode} Visualization"
                            + (f"  ({backend.upper()})" if backend else ""))
        self.resize(1200, 820)
        self.setStyleSheet("QDialog{background:#2b2b2b;} QLabel{color:#ddd;} "
                           "QPushButton{background:#3a3f44;color:#eee;border:none;"
                           "border-radius:4px;padding:6px 12px;} QPushButton:hover{background:#4a5058;}")

        root = QVBoxLayout(self)
        root.addWidget(self._summary_bar())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._playback_tab(), "Playback")
        self.tabs.addTab(self._compare_tab(), "Compare  A | B")
        self.tabs.addTab(self._overview_tab(), "Overview")
        if self.metrics:
            self.tabs.addTab(self._metrics_tab(), "Metrics")
        root.addWidget(self.tabs, 1)

    # ---- summary bar
    def _summary_bar(self):
        bar = QFrame()
        bar.setStyleSheet("QFrame{background:#232629;border-radius:6px;} QLabel{color:#ccc;}")
        lay = QHBoxLayout(bar)
        n_orig = sum(1 for r in self.roles if r.get("role") == "original")
        n_int = len(self.interp_indices)
        lay.addWidget(_badge(f"{len(self.output)} frames", "#34495e"))
        lay.addWidget(_badge(f"{n_orig} original", OK_GREEN))
        lay.addWidget(_badge(f"{n_int} interpolated", ACCENT))
        if self.backend:
            lay.addWidget(_badge(self.backend.upper(), "#8e44ad"))
        if self.metrics:
            ps = np.mean([m[0] for m in self.metrics.values()])
            ss = np.mean([m[1] for m in self.metrics.values()])
            lay.addStretch()
            lay.addWidget(QLabel(f"<b>mean PSNR</b> {ps:.2f} dB &nbsp;&nbsp; <b>mean SSIM</b> {ss:.4f}"))
        lay.addStretch()
        return bar

    # ---- Playback tab
    def _playback_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.pb_pane = ImagePane()
        lay.addWidget(self.pb_pane, 1)

        row = QHBoxLayout()
        self.pb_play_btn = QPushButton("▶ Play")
        self.pb_play_btn.clicked.connect(self._toggle_play)
        row.addWidget(self.pb_play_btn)
        self.pb_only_orig = QCheckBox("Originals only")
        self.pb_only_orig.stateChanged.connect(self._playback_refresh)
        row.addWidget(self.pb_only_orig)
        self.pb_badge = _badge("", "#34495e")
        row.addWidget(self.pb_badge)
        row.addStretch()
        self.pb_info = QLabel("")
        row.addWidget(self.pb_info)
        lay.addLayout(row)

        self.pb_slider = QSlider(Qt.Horizontal)
        self.pb_slider.setMinimum(0)
        self.pb_slider.setMaximum(len(self.output) - 1)
        self.pb_slider.valueChanged.connect(self._playback_show)
        lay.addWidget(self.pb_slider)

        self._play_timer = QTimer(self)
        self._play_timer.setInterval(120)
        self._play_timer.timeout.connect(self._advance)

        self._playback_show(0)
        return w

    def _playback_indices(self):
        if self.pb_only_orig.isChecked():
            return [i for i, r in enumerate(self.roles) if r.get("role") == "original"]
        return list(range(len(self.output)))

    def _playback_refresh(self):
        idxs = self._playback_indices()
        self.pb_slider.setMaximum(max(0, len(idxs) - 1))
        self._playback_show(self.pb_slider.value())

    def _playback_show(self, pos):
        idxs = self._playback_indices()
        if not idxs:
            return
        pos = min(pos, len(idxs) - 1)
        frame_idx = idxs[pos]
        self.pb_pane.set_image(self.output[frame_idx])
        role = self.roles[frame_idx].get("role", "original")
        is_int = role == "interpolated"
        self.pb_badge.setText("INTERPOLATED" if is_int else "ORIGINAL")
        self.pb_badge.setStyleSheet(
            f"color:white; background:{ACCENT if is_int else OK_GREEN}; "
            "border-radius:9px; padding:2px 10px; font-weight:700; font-size:11px;")
        extra = ""
        if frame_idx in self.metrics:
            p, s = self.metrics[frame_idx]
            extra = f"  •  PSNR {p:.2f} dB   SSIM {s:.4f}"
        self.pb_info.setText(f"frame {frame_idx + 1} / {len(self.output)}{extra}")

    def _toggle_play(self):
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.pb_play_btn.setText("▶ Play")
        else:
            self._play_timer.start()
            self.pb_play_btn.setText("⏸ Pause")

    def _advance(self):
        nxt = self.pb_slider.value() + 1
        if nxt > self.pb_slider.maximum():
            nxt = 0
        self.pb_slider.setValue(nxt)

    # ---- Compare tab
    def _compare_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Interpolated frame:"))
        self.cmp_combo = QComboBox()
        for i in self.interp_indices:
            self.cmp_combo.addItem(f"frame {i + 1}", userData=i)
        self.cmp_combo.currentIndexChanged.connect(self._compare_show)
        ctrl.addWidget(self.cmp_combo)

        ctrl.addWidget(QLabel("Reference (B):"))
        self.cmp_ref = QComboBox()
        if self.gt:
            self.cmp_ref.addItem("Ground truth", "gt")
        self.cmp_ref.addItem("Previous original", "prev")
        self.cmp_ref.addItem("Next original", "next")
        self.cmp_ref.currentIndexChanged.connect(self._compare_show)
        ctrl.addWidget(self.cmp_ref)

        self.cmp_diff = QCheckBox("Show difference")
        self.cmp_diff.stateChanged.connect(self._compare_show)
        ctrl.addWidget(self.cmp_diff)

        self.cmp_blink_btn = QPushButton("Blink A/B")
        self.cmp_blink_btn.clicked.connect(self._toggle_blink)
        ctrl.addWidget(self.cmp_blink_btn)

        ctrl.addStretch()
        self.cmp_score = QLabel("")
        self.cmp_score.setStyleSheet(f"color:{ACCENT}; font-weight:700;")
        ctrl.addWidget(self.cmp_score)
        lay.addLayout(ctrl)

        panes = QHBoxLayout()
        self.cmp_a = ImagePane("A - Interpolated")
        self.cmp_b = ImagePane("B - Reference")
        self.cmp_d = ImagePane("Difference", diff=True)
        self.cmp_b.link_to(self.cmp_a)
        self.cmp_d.link_to(self.cmp_a)
        panes.addWidget(self.cmp_a)
        panes.addWidget(self.cmp_b)
        panes.addWidget(self.cmp_d)
        lay.addLayout(panes, 1)

        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(450)
        self._blink_timer.timeout.connect(self._blink_step)
        self._blink_state = False

        if self.interp_indices:
            self._compare_show()
        return w

    def _reference_frame(self, idx):
        ref = self.cmp_ref.currentData()
        if ref == "gt" and idx in self.gt:
            return self.gt[idx], "ground truth"
        if ref == "next" and idx + 1 < len(self.output):
            return self.output[idx + 1], "next original"
        # default / prev
        if idx - 1 >= 0:
            return self.output[idx - 1], "prev original"
        return self.output[idx], "self"

    def _compare_show(self, *_):
        if not self.interp_indices:
            return
        idx = self.cmp_combo.currentData()
        if idx is None:
            idx = self.interp_indices[0]
        a = self.output[idx]
        b, blabel = self._reference_frame(idx)
        self.cmp_a.set_image(a)
        self.cmp_b.set_image(b)
        self.cmp_b.header.setText(f"B - {blabel}")

        self.cmp_d.setVisible(self.cmp_diff.isChecked())
        if self.cmp_diff.isChecked():
            d = np.abs(to_display(a) - to_display(b))
            self.cmp_d.set_image(d)

        # score: prefer stored validation metric, else compute vs current ref
        if idx in self.metrics and self.cmp_ref.currentData() == "gt":
            p, s = self.metrics[idx]
            self.cmp_score.setText(f"PSNR {p:.2f} dB   SSIM {s:.4f}")
        else:
            self.cmp_score.setText(self._quick_scores(a, b))

    def _quick_scores(self, a, b):
        try:
            from skimage.metrics import peak_signal_noise_ratio as psnr
            from skimage.metrics import structural_similarity as ssim
            ga, gb = to_display(a), to_display(b)
            if np.array_equal(ga, gb):
                return "vs B:  identical frames"
            return f"vs B:  PSNR {psnr(gb, ga, data_range=1.0):.2f} dB   SSIM {ssim(gb, ga, data_range=1.0):.4f}"
        except Exception:
            return ""

    def _toggle_blink(self):
        if self._blink_timer.isActive():
            self._blink_timer.stop()
            self.cmp_blink_btn.setText("Blink A/B")
            self._compare_show()
        else:
            self._blink_timer.start()
            self.cmp_blink_btn.setText("Stop blink")

    def _blink_step(self):
        idx = self.cmp_combo.currentData()
        if idx is None:
            return
        self._blink_state = not self._blink_state
        if self._blink_state:
            b, blabel = self._reference_frame(idx)
            self.cmp_a.set_image(b)
            self.cmp_a.header.setText(f"A - showing B ({blabel})")
        else:
            self.cmp_a.set_image(self.output[idx])
            self.cmp_a.header.setText("A - Interpolated")

    # ---- Overview tab
    def _overview_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setSpacing(6)
        cols = 6
        for n, i in enumerate(range(len(self.output))):
            is_int = self.roles[i].get("role") == "interpolated"
            cell = QVBoxLayout()
            thumb = QLabel()
            thumb.setPixmap(to_qpixmap(self.output[i], max_side=170))
            thumb.setAlignment(Qt.AlignCenter)
            border = ACCENT if is_int else "#3a3a3a"
            thumb.setStyleSheet(f"border:2px solid {border}; background:{BG};")
            thumb.setCursor(Qt.PointingHandCursor)
            thumb.mousePressEvent = (lambda ev, k=i: self._jump_to(k))
            cap = QLabel(f"{i + 1}" + ("  ●" if is_int else ""))
            cap.setAlignment(Qt.AlignCenter)
            cap.setStyleSheet(f"color:{ACCENT if is_int else '#aaa'}; font-size:11px;")
            cell.addWidget(thumb)
            cell.addWidget(cap)
            cw = QWidget()
            cw.setLayout(cell)
            grid.addWidget(cw, n // cols, n % cols)
        scroll.setWidget(holder)
        return scroll

    def _jump_to(self, frame_idx):
        self.pb_only_orig.setChecked(False)
        self.pb_slider.setValue(frame_idx)
        self.tabs.setCurrentIndex(0)

    # ---- Metrics tab (validation only)
    def _metrics_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        items = sorted(self.metrics.items())
        table = QTableWidget(len(items), 3)
        table.setHorizontalHeaderLabels(["Frame", "PSNR (dB)", "SSIM"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for r, (idx, (p, s)) in enumerate(items):
            table.setItem(r, 0, QTableWidgetItem(str(idx + 1)))
            table.setItem(r, 1, QTableWidgetItem(f"{p:.2f}"))
            table.setItem(r, 2, QTableWidgetItem(f"{s:.4f}"))
        lay.addWidget(table, 1)

        if _HAVE_PG and items:
            xs = [idx + 1 for idx, _ in items]
            ps = [m[0] for _, m in items]
            ss = [m[1] for _, m in items]
            p_plot = pg.PlotWidget(title="PSNR per interpolated frame (dB)")
            p_plot.setBackground(BG)
            p_plot.showGrid(x=True, y=True, alpha=0.3)
            p_plot.plot(xs, ps, pen=pg.mkPen(ACCENT, width=2), symbol="o", symbolSize=5)
            lay.addWidget(p_plot, 1)
            s_plot = pg.PlotWidget(title="SSIM per interpolated frame")
            s_plot.setBackground(BG)
            s_plot.showGrid(x=True, y=True, alpha=0.3)
            s_plot.plot(xs, ss, pen=pg.mkPen(OK_GREEN, width=2), symbol="o", symbolSize=5)
            s_plot.setXLink(p_plot)
            lay.addWidget(s_plot, 1)
        return w


# --------------------------------------------------------- standalone self-test
if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    rng = np.random.default_rng(0)
    M, H, W = 9, 96, 128
    out = (rng.random((M, H, W)) * 255).astype(np.uint8)
    roles = [{"role": "interpolated" if i % 2 else "original"} for i in range(M)]
    gt = {i: (out[i].astype(np.float32) + rng.normal(0, 12, (H, W))).clip(0, 255).astype(np.uint8)
          for i in range(M) if i % 2}
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    metrics = {i: (psnr(to_display(gt[i]), to_display(out[i]), data_range=1.0),
                   ssim(to_display(gt[i]), to_display(out[i]), data_range=1.0)) for i in gt}
    app = QApplication(sys.argv)
    dlg = InterpolationVisualizer(out, roles, gt=gt, metrics=metrics, backend="rife")
    dlg.show()
    sys.exit(app.exec())
