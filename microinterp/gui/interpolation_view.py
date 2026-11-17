"""Qt widget to load a series, interpolate it with any backend, view the result
and validate it against held-out frames.
"""

import os
import numpy as np
import tifffile
from PIL import Image
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QProgressBar, QSlider, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from .interpolation_dialog import InterpolationDialog
from .. import core as interpolate
from ..backends import check, BACKENDS
from .styles import *


def _missing_weights_message(backend, reason):
    """Human-friendly message when a backend cannot run here."""
    return f"{BACKENDS[backend]['name']} is unavailable.\n\n{reason}"


class InterpolationWorker(QThread):
    """Worker thread for running interpolation without blocking UI"""
    finished = Signal(np.ndarray, list, str)
    progress = Signal(str)
    error = Signal(str)
    
    def __init__(self, images, intervals, output_path=None, save_as_folder=False, save_ext='.tif', backend='rife'):
        super().__init__()
        self.images = images
        self.intervals = intervals
        self.output_path = output_path
        self.save_as_folder = save_as_folder
        self.save_ext = save_ext
        self.backend = backend

    def run(self):
        try:
            model_name = BACKENDS.get(self.backend, {}).get('name', self.backend.upper())
            self.progress.emit(f"Loading {model_name} model...")

            ok, reason = check(self.backend)
            if not ok:
                self.error.emit(_missing_weights_message(self.backend, reason))
                return

            self.progress.emit(f"Interpolating {len(self.images)} frames with {len(self.intervals)} interval(s) using {model_name}...")

            # Run interpolation
            interpolated_images, new_intervals = interpolate.interpolate_intervals(
                self.images, self.intervals, backend=self.backend
            )
            
            self.progress.emit(f"Interpolation complete! Generated {len(interpolated_images)} frames.")
            
            # Save if output path is provided
            if self.output_path:
                if self.save_as_folder:
                    # Save as individual TIFF files in folder
                    self.progress.emit(f"Saving {len(interpolated_images)} frames to folder...")
                    
                    # Create output directory if it doesn't exist
                    os.makedirs(self.output_path, exist_ok=True)
                    
                    # Determine padding width based on number of frames
                    padding = len(str(len(interpolated_images)))
                    
                    # Save each frame
                    for i, frame in enumerate(interpolated_images):
                        filename = f"frame_{str(i).zfill(padding)}{self.save_ext}"
                        filepath = os.path.join(self.output_path, filename)
                        # Save PNG via PIL for non-tif, otherwise tifffile
                        if self.save_ext.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            from PIL import Image
                            im = Image.fromarray(frame.astype(np.uint8))
                            im.save(filepath)
                        else:
                            tifffile.imwrite(filepath, frame)
                        
                        if (i + 1) % 10 == 0:
                            self.progress.emit(f"  Saved {i + 1}/{len(interpolated_images)} frames...")
                    
                    self.progress.emit(f"✓ Saved {len(interpolated_images)} frames to folder!")
                else:
                    # Save as single multi-frame TIFF
                    self.progress.emit(f"Saving to {self.output_path}...")
                    tifffile.imwrite(self.output_path, interpolated_images)
                    self.progress.emit("Save complete!")
            
            self.finished.emit(interpolated_images, new_intervals, self.output_path or "")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.error.emit(f"Interpolation error: {str(e)}\n\nDetails:\n{error_details}")


class ValidationWorker(QThread):
    """Reconstruct every other real frame with 2x interpolation and score it
    (PSNR/SSIM). Emits (output_stack, roles, gt_dict, metrics_dict).
    """
    # `object`: PySide6 cannot marshal a bare dict through a signal
    finished = Signal(object, object, object, object, str)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, images, backend='rife'):
        super().__init__()
        self.images = images
        self.backend = backend

    def run(self):
        try:
            ok, reason = check(self.backend)
            if not ok:
                self.error.emit(_missing_weights_message(self.backend, reason))
                return
            if len(self.images) < 3:
                self.error.emit("Need at least 3 frames to validate.")
                return
            self.progress.emit(f"Validating with {self.backend.upper()} (reconstructing held-out frames)...")
            inp = self.images[::2]
            intervals = [{"start": 0, "stop": len(inp) - 1, "interp": 1}]
            out, _ = interpolate.interpolate_intervals(inp, intervals, backend=self.backend)
            roles = interpolate.describe_layout(len(inp), intervals)

            from skimage.metrics import peak_signal_noise_ratio as psnr
            from skimage.metrics import structural_similarity as ssim
            gt, metrics = {}, {}
            for i, role in enumerate(roles):
                if role.get("role") == "interpolated" and i < len(self.images):
                    truth = self.images[i]
                    gt[i] = truth
                    lo, hi = float(truth.min()), float(truth.max())
                    rng = (hi - lo) or 1.0
                    g = np.clip((truth.astype(np.float32) - lo) / rng, 0, 1)
                    p = np.clip((out[i].astype(np.float32) - lo) / rng, 0, 1)
                    metrics[i] = (float(psnr(g, p, data_range=1.0)),
                                  float(ssim(g, p, data_range=1.0)))
            self.finished.emit(out, roles, gt, metrics, self.backend)
        except Exception as e:
            import traceback
            self.error.emit(f"Validation error: {str(e)}\n\nDetails:\n{traceback.format_exc()}")


class InterpolationView(QWidget):
    """View for configuring and running RIFE frame interpolation"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Enable keyboard focus for arrow key navigation
        from PySide6.QtCore import Qt
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        self.images = None
        self.image_path = None
        self.interpolated_images = None
        self.intervals = None
        self.roles = None
        self.worker = None
        self.validation_worker = None
        self._visualizers = []  # keep references so windows aren't GC'd
        self.current_frame = 0
        self.viewing_interpolated = False
        self.backend = 'rife'  # 'rife' (default), 'film', 'cdfi', or 'ldmvfi'

        self.init_ui()
    
    def keyPressEvent(self, event):
        """Handle keyboard events for frame navigation"""
        if event.key() == Qt.Key_Left:
            # Left arrow - previous frame
            if self.prev_frame_btn.isEnabled():
                self.prev_frame()
        elif event.key() == Qt.Key_Right:
            # Right arrow - next frame
            if self.next_frame_btn.isEnabled():
                self.next_frame()
        else:
            super().keyPressEvent(event)
        
    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)
        
        # Top control panel
        control_panel = QWidget()
        control_panel.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['surface']};
                border-bottom: 1px solid {COLORS['border']};
            }}
        """)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(12, 10, 12, 10)
        control_layout.setSpacing(8)
        
        # Title row
        title_row = QHBoxLayout()
        title = QLabel("Frame Interpolation")
        title.setStyleSheet(get_label_style(size='title', weight='bold', color='primary'))
        title_row.addWidget(title)
        
        # Info labels in title row
        self.sequence_info_label = QLabel("No sequence loaded")
        self.sequence_info_label.setStyleSheet(get_label_style(size='small', color='light_darker'))
        title_row.addWidget(self.sequence_info_label)
        
        self.intervals_info_label = QLabel("No intervals set")
        self.intervals_info_label.setStyleSheet(get_label_style(size='small', color='light_darker'))
        title_row.addWidget(self.intervals_info_label)
        
        title_row.addStretch()
        control_layout.addLayout(title_row)
        
        # Main controls row
        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)

        left_group = QHBoxLayout()
        left_group.setSpacing(8)

        self.load_sequence_btn = QPushButton("Load Series")
        self.load_sequence_btn.setStyleSheet(get_button_style('primary'))
        self.load_sequence_btn.clicked.connect(self.load_sequence)
        left_group.addWidget(self.load_sequence_btn)
        
        self.configure_btn = QPushButton("Set Intervals")
        self.configure_btn.setStyleSheet(get_button_style('secondary'))
        self.configure_btn.setEnabled(False)
        self.configure_btn.clicked.connect(self.configure_interpolation)
        left_group.addWidget(self.configure_btn)

        # Backend selector: choose the interpolation model.
        model_label = QLabel("Model:")
        model_label.setStyleSheet(get_label_style(size='small'))
        left_group.addWidget(model_label)
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet(f"""
            QComboBox {{ background-color: {COLORS['primary']}; color: white; border: none;
                         padding: 5px 10px; border-radius: 2px; font-weight: 500; font-size: 11px; }}
            QComboBox QAbstractItemView {{ background-color: {COLORS['surface']}; color: white;
                         selection-background-color: {COLORS['primary']}; }}
        """)
        for b in BACKENDS:  # rife, film, cdfi, ldmvfi
            meta = BACKENDS[b]
            self.model_combo.addItem(meta['name'], userData=b)
        self.model_combo.setToolTip(
            "Interpolation model:\n"
            "RIFE - fast, CPU or GPU (default)\n"
            "FILM - strong on large motion, GPU\n"
            "CDFI - compact & sharp, GPU only\n"
            "LDMVFI - highest quality, slowest, GPU only"
        )
        self.model_combo.currentIndexChanged.connect(self.on_backend_changed)
        left_group.addWidget(self.model_combo)

        center_group = QHBoxLayout()
        center_group.setSpacing(8)
        self.run_btn = QPushButton("▶ Run Interpolation")
        self.run_btn.setStyleSheet(get_button_style('success'))
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_interpolation)
        center_group.addWidget(self.run_btn)

        right_group = QHBoxLayout()
        right_group.setSpacing(8)
        self.save_hybrid_btn = QPushButton("Save Hybrid")
        self.save_hybrid_btn.setStyleSheet(get_button_style('secondary'))
        self.save_hybrid_btn.setEnabled(False)
        self.save_hybrid_btn.clicked.connect(self.save_hybrid_series)
        right_group.addWidget(self.save_hybrid_btn)

        self.view_original_btn = QPushButton("Original")
        self.view_original_btn.setCheckable(True)
        self.view_original_btn.setChecked(True)
        self.view_original_btn.setEnabled(False)
        self.view_original_btn.setStyleSheet(get_button_style('secondary'))
        self.view_original_btn.clicked.connect(self.show_original_images)
        right_group.addWidget(self.view_original_btn)
        
        self.view_interpolated_btn = QPushButton("Hybrid")
        self.view_interpolated_btn.setCheckable(True)
        self.view_interpolated_btn.setEnabled(False)
        self.view_interpolated_btn.setStyleSheet(get_button_style('secondary'))
        self.view_interpolated_btn.clicked.connect(self.show_interpolated_images)
        self.view_interpolated_btn.setToolTip("View complete hybrid sequence (all original + interpolated frames)")
        right_group.addWidget(self.view_interpolated_btn)

        self.visualize_btn = QPushButton("📊 Visualize")
        self.visualize_btn.setStyleSheet(get_button_style('primary'))
        self.visualize_btn.setEnabled(False)
        self.visualize_btn.setToolTip("Open the result visualizer: playback, side-by-side compare, overview")
        self.visualize_btn.clicked.connect(self.open_visualizer)
        right_group.addWidget(self.visualize_btn)

        self.validate_btn = QPushButton("✓ Validate")
        self.validate_btn.setStyleSheet(get_button_style('secondary'))
        self.validate_btn.setEnabled(False)
        self.validate_btn.setToolTip(
            "Quality check: drop every other real frame, reconstruct it with the\n"
            "selected model, and score it (PSNR/SSIM) against the held-out original.")
        self.validate_btn.clicked.connect(self.run_validation)
        right_group.addWidget(self.validate_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(get_progressbar_style())
        right_group.addWidget(self.progress_bar)

        controls_row.addLayout(left_group)
        controls_row.addStretch()
        controls_row.addLayout(center_group)
        controls_row.addStretch()
        controls_row.addLayout(right_group)

        control_layout.addLayout(controls_row)
        
        main_layout.addWidget(control_panel)
        
        # Main canvas area
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout()
        canvas_layout.setContentsMargins(20, 20, 20, 20)
        canvas_layout.setSpacing(0)
        canvas_widget.setLayout(canvas_layout)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['background']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
            }}
        """)
        self.image_label.setText(
            f"<div style='text-align:center;'>"
            f"<div style='font-size:28px; color:{COLORS['light_darker']};'>🖼</div>"
            f"<div style='font-size:16px; color:{COLORS['light']}; font-weight:600;'>No sequence loaded</div>"
            f"<div style='font-size:12px; color:{COLORS['light_darker']};'>Load a series to preview frames.</div>"
            f"<div style='font-size:11px; color:{COLORS['light_darker']};'>Shortcut: Ctrl+O</div>"
            f"</div>"
        )
        canvas_layout.addWidget(self.image_label, 1)
        
        main_layout.addWidget(canvas_widget, 1)
        
        # Frame navigation at bottom
        nav_widget = QWidget()
        nav_widget.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border-top: 1px solid #3a3a3a;
            }
        """)
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(20, 10, 20, 10)
        
        self.prev_frame_btn = QPushButton("◀")
        self.prev_frame_btn.setStyleSheet(get_button_style('secondary'))
        self.prev_frame_btn.setFixedWidth(50)
        self.prev_frame_btn.setEnabled(False)
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        nav_layout.addWidget(self.prev_frame_btn)
        
        self.frame_info_label = QLabel("Frame: 0 / 0")
        self.frame_info_label.setStyleSheet(get_label_style(size='small'))
        self.frame_info_label.setFixedWidth(100)
        self.frame_info_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.frame_info_label)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.update_displayed_frame)
        self.frame_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #3a3a3a;
                height: 6px;
                background: #1e1e1e;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #007acc;
                border: 1px solid #005a9e;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #1e8ad6;
            }
        """)
        nav_layout.addWidget(self.frame_slider, 1)
        
        self.next_frame_btn = QPushButton("▶")
        self.next_frame_btn.setStyleSheet(get_button_style('secondary'))
        self.next_frame_btn.setFixedWidth(50)
        self.next_frame_btn.setEnabled(False)
        self.next_frame_btn.clicked.connect(self.next_frame)
        nav_layout.addWidget(self.next_frame_btn)
        
        main_layout.addWidget(nav_widget)
    
    def get_group_style(self):
        return get_groupbox_style()
    
    def get_button_style(self, bg_color="primary", hover_color=None):
        color = bg_color if bg_color in COLORS else 'primary'
        return get_button_style(color, hover_color)
    
    def get_toggle_button_style(self):
        return f"""
            QPushButton {{
                background-color: {COLORS['dark_lighter']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 18px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {COLORS['dark']};
            }}
            QPushButton:checked {{
                background-color: {COLORS['primary']};
                border: 2px solid {COLORS['primary_dark']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['surface']};
                color: {COLORS['light_darker']};
            }}
        """
    
    def normalize_image_for_display(self, img):
        """Normalize image to 0-255 range for display"""
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
        return img
    
    def display_frame(self, frame_data):
        """Display a single frame in the image viewer"""
        if frame_data is None:
            self.image_label.setText("No image to display")
            return
        
        # Normalize to 0-255
        normalized = self.normalize_image_for_display(frame_data)
        
        # Create QImage
        height, width = normalized.shape
        bytes_per_line = width
        q_image = QImage(normalized.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        # Convert to pixmap
        pixmap = QPixmap.fromImage(q_image)
        
        # cap the preview at 800 x 800
        max_width = 800
        max_height = 800
        
        # Scale to fit within max bounds while maintaining aspect ratio
        if pixmap.width() > max_width or pixmap.height() > max_height:
            scaled_pixmap = pixmap.scaled(
                max_width,
                max_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        else:
            scaled_pixmap = pixmap
        
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)
    
    def update_displayed_frame(self):
        """Update the displayed frame based on slider position"""
        if not hasattr(self, 'viewing_interpolated'):
            return
            
        self.current_frame = self.frame_slider.value()
        
        if self.viewing_interpolated and self.interpolated_images is not None:
            if 0 <= self.current_frame < len(self.interpolated_images):
                self.display_frame(self.interpolated_images[self.current_frame])
                self.frame_info_label.setText(
                    f"{self.current_frame + 1}/{len(self.interpolated_images)} • Hybrid Sequence"
                )
        elif self.images is not None:
            if 0 <= self.current_frame < len(self.images):
                self.display_frame(self.images[self.current_frame])
                self.frame_info_label.setText(
                    f"{self.current_frame + 1}/{len(self.images)} • Original"
                )
    
    def show_original_images(self):
        """Switch to viewing original images"""
        if not self.view_original_btn.isChecked():
            self.view_original_btn.setChecked(True)
            return
            
        self.viewing_interpolated = False
        self.view_interpolated_btn.setChecked(False)
        
        if self.images is not None:
            self.frame_slider.setMaximum(len(self.images) - 1)
            self.current_frame = min(self.current_frame, len(self.images) - 1)
            self.frame_slider.setValue(self.current_frame)
            self.update_displayed_frame()
    
    def show_interpolated_images(self):
        """Switch to viewing interpolated images"""
        if not self.view_interpolated_btn.isChecked():
            self.view_interpolated_btn.setChecked(True)
            return
            
        if self.interpolated_images is None:
            self.view_interpolated_btn.setChecked(False)
            QMessageBox.warning(self, "No Results", "No interpolated images available. Please run interpolation first.")
            return
            
        self.viewing_interpolated = True
        self.view_original_btn.setChecked(False)
        
        self.frame_slider.setMaximum(len(self.interpolated_images) - 1)
        self.current_frame = min(self.current_frame, len(self.interpolated_images) - 1)
        self.frame_slider.setValue(self.current_frame)
        self.update_displayed_frame()
    
    def prev_frame(self):
        """Go to previous frame"""
        if self.current_frame > 0:
            self.frame_slider.setValue(self.current_frame - 1)
    
    def next_frame(self):
        """Go to next frame"""
        max_frame = self.frame_slider.maximum()
        if self.current_frame < max_frame:
            self.frame_slider.setValue(self.current_frame + 1)
    
    def log(self, message):
        """Add message to log (disabled - status view removed)"""
        pass  # Status log removed from UI
        
    def load_sequence(self):
        """Load a TIFF sequence from folder or multi-frame file"""
        # Ask user to choose between folder or file
        choice = QMessageBox.question(
            self,
            "Load Time Series",
            "Load from folder of TIFF images?\n\nYes = Folder of TIFFs\nNo = Single multi-frame TIFF",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if choice == QMessageBox.Yes:
            # Load from folder
            folder_path = QFileDialog.getExistingDirectory(
                self,
                "Select Folder with TIFF Sequence",
                ""
            )
            
            if folder_path:
                try:
                    self.log(f"Loading from folder: {os.path.basename(folder_path)}")
                    
                    # Get all image files in the folder (TIFF, PNG, JPG, BMP)
                    valid_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
                    tiff_files = sorted([
                        f for f in os.listdir(folder_path)
                        if f.lower().endswith(valid_exts)
                    ])
                    
                    if not tiff_files:
                        QMessageBox.warning(self, "No TIFFs Found", 
                                          "No TIFF files found in the selected folder.")
                        self.log("ERROR: No TIFF files found")
                        return
                    
                    self.log(f"Found {len(tiff_files)} TIFF files")
                    
                    # Load all images
                    images_list = []
                    def _read_image_as_gray(path):
                        """Read image at path and return a 2D numpy grayscale array.
                        Uses tifffile for TIFFs (supports multi-page) and PIL for other formats."""
                        lp = str(path)
                        ext = lp.lower()
                        if ext.endswith(('.tif', '.tiff')):
                            im = tifffile.imread(lp)
                            # If multi-page/tiff -> take first page
                            if im.ndim == 3:
                                # If color RGB stack, convert to first page
                                if im.shape[0] in (3,) and im.shape[-1] in (3,):
                                    # ambiguous shape, fall back to PIL
                                    im = Image.open(lp)
                                    im = im.convert('L')
                                    return np.array(im)
                                else:
                                    im = im[0]
                            # If RGB image, convert to grayscale
                            if im.ndim == 3:
                                # assume HxWxC
                                im = np.asarray(Image.fromarray(im).convert('L'))
                            return im
                        else:
                            im = Image.open(lp).convert('L')
                            return np.array(im)

                    for i, filename in enumerate(tiff_files):
                        file_path = os.path.join(folder_path, filename)
                        img = _read_image_as_gray(file_path)
                        images_list.append(img)
                        
                        if (i + 1) % 10 == 0:
                            self.log(f"  Loaded {i + 1}/{len(tiff_files)} images...")
                    
                    # Stack into 3D array (T, H, W)
                    self.images = np.stack(images_list, axis=0)
                    self.image_path = folder_path
                    
                    num_frames = self.images.shape[0]
                    height, width = self.images.shape[1], self.images.shape[2]
                    
                    info_text = f"{num_frames} frames • {height}×{width}px"
                    self.sequence_info_label.setText(info_text)
                    self.log(f"Loaded: {info_text}")
                    
                    # Enable configuration
                    self.configure_btn.setEnabled(True)
                    self.validate_btn.setEnabled(True)
                    self.intervals = None
                    self.intervals_info_label.setText("No intervals set")
                    self.run_btn.setEnabled(False)
                    
                    # Enable viewer controls
                    self.view_original_btn.setEnabled(True)
                    self.frame_slider.setEnabled(True)
                    self.frame_slider.setMaximum(num_frames - 1)
                    self.frame_slider.setValue(0)
                    self.prev_frame_btn.setEnabled(True)
                    self.next_frame_btn.setEnabled(True)
                    
                    # Show first frame
                    self.current_frame = 0
                    self.viewing_interpolated = False
                    self.update_displayed_frame()
                    
                except Exception as e:
                    QMessageBox.critical(self, "Load Error", f"Failed to load folder:\n{str(e)}")
                    self.log(f"ERROR: {str(e)}")
        else:
            # Load from single file
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Image Sequence",
                "",
                "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)"
            )
            
            if file_path:
                try:
                    self.log(f"Loading: {os.path.basename(file_path)}")
                    # Try tifffile first for multi-frame TIFFs; fall back to PIL for other formats
                    try:
                        self.images = tifffile.imread(file_path)
                    except Exception:
                        # Load single/multi-image via PIL
                        with Image.open(file_path) as imf:
                            frames = []
                            try:
                                i = 0
                                while True:
                                    imf.seek(i)
                                    frames.append(np.array(imf.convert('L')))
                                    i += 1
                            except EOFError:
                                pass
                            if len(frames) == 1:
                                self.images = np.expand_dims(frames[0], axis=0)
                            else:
                                self.images = np.stack(frames, axis=0)
                    
                    # Ensure 3D array (T, H, W)
                    if self.images.ndim == 2:
                        self.images = np.expand_dims(self.images, axis=0)
                    
                    self.image_path = file_path
                    num_frames = self.images.shape[0]
                    height, width = self.images.shape[1], self.images.shape[2]
                    
                    info_text = f"{num_frames} frames • {height}×{width}px"
                    self.sequence_info_label.setText(info_text)
                    self.log(f"Loaded: {info_text}")
                    
                    # Enable configuration
                    self.configure_btn.setEnabled(True)
                    self.validate_btn.setEnabled(True)
                    self.intervals = None
                    self.intervals_info_label.setText("No intervals set")
                    self.run_btn.setEnabled(False)
                    
                    # Enable viewer controls
                    self.view_original_btn.setEnabled(True)
                    self.frame_slider.setEnabled(True)
                    self.frame_slider.setMaximum(num_frames - 1)
                    self.frame_slider.setValue(0)
                    self.prev_frame_btn.setEnabled(True)
                    self.next_frame_btn.setEnabled(True)
                    
                    # Show first frame
                    self.current_frame = 0
                    self.viewing_interpolated = False
                    self.update_displayed_frame()
                    
                except Exception as e:
                    QMessageBox.critical(self, "Load Error", f"Failed to load sequence:\n{str(e)}")
                    self.log(f"ERROR: {str(e)}")
    
    def on_backend_changed(self, _index):
        """Update the active backend from the model selector."""
        self.backend = self.model_combo.currentData() or 'rife'
        self.log(f"Interpolation model set to {self.backend.upper()}")

    def configure_interpolation(self):
        """Open dialog to configure interpolation intervals"""
        if self.images is None:
            return
            
        num_frames = self.images.shape[0]
        dialog = InterpolationDialog(num_frames, self)
        
        if dialog.exec():
            self.intervals = dialog.get_intervals()
            
            # Calculate expected output frames
            expected_frames = interpolate.get_interpolated_length(num_frames, self.intervals)
            
            info_text = f"Configured {len(self.intervals)} interval(s), will generate {expected_frames} frames"
            self.intervals_info_label.setText(info_text)
            self.log(info_text)
            
            # Log details
            for i, interval in enumerate(self.intervals):
                factor = 2 ** interval['interp']
                self.log(f"  Interval {i+1}: frames {interval['start']}-{interval['stop']}, {factor}x interpolation")
            
            # Enable run button
            self.run_btn.setEnabled(True)
    
    def run_interpolation(self):
        """Run the interpolation process"""
        if self.images is None or self.intervals is None:
            return
        
        # Run interpolation without saving (user can save hybrid series separately)
        output_path = None
        save_as_folder = False
        
        # Disable buttons during processing
        self.run_btn.setEnabled(False)
        self.configure_btn.setEnabled(False)
        self.load_sequence_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        
        self.log("Starting interpolation...")
        
        # Determine preferred save extension based on input
        save_ext = '.tif'
        try:
            if os.path.isdir(self.image_path):
                # look at first image file in folder
                for f in os.listdir(self.image_path):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        save_ext = os.path.splitext(f)[1]
                        break
            else:
                _, ext = os.path.splitext(self.image_path)
                if ext.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                    save_ext = ext
        except Exception:
            pass

        # Create and start worker thread
        self.worker = InterpolationWorker(self.images, self.intervals, output_path, save_as_folder, save_ext=save_ext, backend=self.backend)
        self.worker.progress.connect(self.log)
        self.worker.finished.connect(self.on_interpolation_finished)
        self.worker.error.connect(self.on_interpolation_error)
        self.worker.start()
    
    def on_interpolation_finished(self, interpolated_images, new_intervals, output_path):
        """Handle successful interpolation"""
        self.interpolated_images = interpolated_images
        self.progress_bar.setVisible(False)
        
        # Calculate number of new interpolated frames
        num_interpolated = len(interpolated_images) - len(self.images)
        
        self.log(f"✓ Interpolation complete: {len(interpolated_images)} total frames (hybrid sequence)")
        
        # Per-frame roles (original vs interpolated) for the visualizer.
        try:
            self.roles = interpolate.describe_layout(len(self.images), self.intervals)
        except Exception:
            self.roles = [{"role": "original"} for _ in range(len(interpolated_images))]

        # Enable interpolated view button, save hybrid button and the visualizer.
        self.view_interpolated_btn.setEnabled(True)
        self.save_hybrid_btn.setEnabled(True)
        self.visualize_btn.setEnabled(True)
        
        # Automatically switch to interpolated view (which now shows the complete hybrid sequence)
        self.view_interpolated_btn.setChecked(True)
        self.view_original_btn.setChecked(False)
        self.show_interpolated_images()
        
        QMessageBox.information(
            self,
            "Success",
            f"Hybrid sequence complete!\n\n"
            f"Total frames: {len(interpolated_images)}\n"
            f"Original frames: {len(self.images)}\n"
            f"New interpolated frames: {num_interpolated}\n\n"
            f"The viewer now shows the complete hybrid sequence with all frames in temporal order.\n"
            f"Click '📊 Visualize' to inspect / compare frames, or 'Save Hybrid' to export."
        )
        
        # Re-enable buttons
        self.run_btn.setEnabled(True)
        self.configure_btn.setEnabled(True)
        self.load_sequence_btn.setEnabled(True)
    
    def on_interpolation_error(self, error_message):
        """Handle interpolation error"""
        self.progress_bar.setVisible(False)
        self.log(f"✗ ERROR: {error_message}")
        QMessageBox.critical(self, "Interpolation Error", error_message)
        
        # Re-enable buttons
        self.run_btn.setEnabled(True)
        self.configure_btn.setEnabled(True)
        self.load_sequence_btn.setEnabled(True)

    # ------------------------------------------------------------- visualization
    def _open_visualizer(self, output, roles, gt=None, metrics=None):
        from .visualization import InterpolationVisualizer
        viz = InterpolationVisualizer(output, roles, gt=gt, metrics=metrics,
                                      backend=self.backend, parent=self)
        self._visualizers.append(viz)  # keep a reference
        viz.show()

    def open_visualizer(self):
        """Open the result visualizer over the current hybrid sequence."""
        if self.interpolated_images is None:
            QMessageBox.warning(self, "No Results", "Run interpolation first.")
            return
        roles = self.roles
        if not roles or len(roles) != len(self.interpolated_images):
            roles = [{"role": "original"} for _ in range(len(self.interpolated_images))]
        self._open_visualizer(self.interpolated_images, roles)

    def run_validation(self):
        """Reconstruct held-out frames with the selected model and score them."""
        if self.images is None:
            QMessageBox.warning(self, "No Data", "Load a sequence first.")
            return
        if len(self.images) < 3:
            QMessageBox.warning(self, "Too Few Frames", "Validation needs at least 3 frames.")
            return
        self.validate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log(f"Validating with {self.backend.upper()}...")
        self.validation_worker = ValidationWorker(self.images, backend=self.backend)
        self.validation_worker.progress.connect(self.log)
        self.validation_worker.finished.connect(self.on_validation_finished)
        self.validation_worker.error.connect(self.on_validation_error)
        self.validation_worker.start()

    def on_validation_finished(self, output, roles, gt, metrics, backend):
        self.progress_bar.setVisible(False)
        self.validate_btn.setEnabled(True)
        if metrics:
            mp = float(np.mean([m[0] for m in metrics.values()]))
            ms = float(np.mean([m[1] for m in metrics.values()]))
            self.log(f"✓ Validation: mean PSNR={mp:.2f} dB, SSIM={ms:.4f} over {len(metrics)} frames")
        self._open_visualizer(output, roles, gt=gt, metrics=metrics)

    def on_validation_error(self, message):
        self.progress_bar.setVisible(False)
        self.validate_btn.setEnabled(True)
        QMessageBox.critical(self, "Validation Error", message)

    def save_hybrid_series(self):
        """Save hybrid time series (real + interpolated frames in correct temporal order)"""
        if self.images is None or self.interpolated_images is None or self.intervals is None:
            QMessageBox.warning(self, "No Data", "Please run interpolation first.")
            return
        
        # Determine base directory
        if os.path.isdir(self.image_path):
            base_dir = os.path.dirname(self.image_path)
            folder_name = os.path.basename(self.image_path)
        else:
            base_dir = os.path.dirname(self.image_path)
            folder_name = os.path.splitext(os.path.basename(self.image_path))[0]
        
        # Create default output path as "hybrid" folder
        default_path = os.path.join(base_dir, f"{folder_name}_hybrid")
        
        output_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Hybrid Series Output Folder",
            default_path
        )
        
        if not output_folder:
            return
        
        try:
            # Create output directory
            os.makedirs(output_folder, exist_ok=True)
            
            self.log("Saving hybrid time series...")
            
            # originals + interpolated frames, in temporal order
            hybrid_frames = self.interpolated_images
            
            # Determine file extension and padding
            save_ext = '.tif'
            try:
                if os.path.isdir(self.image_path):
                    for f in os.listdir(self.image_path):
                        if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')):
                            save_ext = os.path.splitext(f)[1]
                            break
                else:
                    _, ext = os.path.splitext(self.image_path)
                    if ext.lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'):
                        save_ext = ext
            except Exception:
                pass
            
            padding = len(str(len(hybrid_frames)))
            
            # Calculate number of interpolated frames
            num_interpolated = len(hybrid_frames) - len(self.images)
            
            # Save hybrid frames
            self.log(f"Saving {len(hybrid_frames)} frames ({len(self.images)} original + {num_interpolated} interpolated)...")
            
            for i, frame in enumerate(hybrid_frames):
                filename = f"frame_{str(i).zfill(padding)}{save_ext}"
                filepath = os.path.join(output_folder, filename)
                
                # Save based on format
                if save_ext.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                    from PIL import Image
                    im = Image.fromarray(frame.astype(np.uint8))
                    im.save(filepath)
                else:
                    tifffile.imwrite(filepath, frame)
                
                if (i + 1) % 20 == 0:
                    self.log(f"  Saved {i + 1}/{len(hybrid_frames)} frames...")
            
            self.log(f"✓ Hybrid series saved to: {output_folder}")
            
            QMessageBox.information(
                self,
                "Success",
                f"Hybrid time series saved successfully!\n\n"
                f"Total frames: {len(hybrid_frames)}\n"
                f"Original frames: {len(self.images)}\n"
                f"Interpolated frames: {num_interpolated}\n\n"
                f"Saved to:\n{output_folder}"
            )
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.log(f"✗ ERROR: {str(e)}")
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save hybrid series:\n{str(e)}\n\nDetails:\n{error_details}"
            )
