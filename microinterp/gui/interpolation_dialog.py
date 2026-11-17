import math
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSpinBox, QComboBox, QFrame, QDialogButtonBox
)
from PySide6.QtCore import Signal


class InterpolationIntervalWidget(QFrame):
    """Widget for configuring a single interpolation interval"""
    removed = Signal(object)
    
    def __init__(self, parent=None, max_frame=10000):
        super().__init__(parent)
        
        self.start_frame_spinbox = QSpinBox()
        self.start_frame_spinbox.setRange(0, max_frame - 1)
        
        self.end_frame_spinbox = QSpinBox()
        self.end_frame_spinbox.setRange(1, max_frame)
        
        self.end_frame_spinbox.valueChanged.connect(self.update_start_frame_max)
        
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(["2x", "4x", "8x", "16x"])
        self.interpolation_combo.setMinimumWidth(80)
        
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_interval)
        
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Start Frame:"))
        layout.addWidget(self.start_frame_spinbox)
        layout.addWidget(QLabel("End Frame:"))
        layout.addWidget(self.end_frame_spinbox)
        layout.addWidget(QLabel("Interpolation:"))
        layout.addWidget(self.interpolation_combo)
        layout.addWidget(self.remove_button)
        
        self.setLayout(layout)
        
        # Apply styling
        self.setStyleSheet("""
            QFrame {
                background-color: #34495e;
                border-radius: 4px;
                padding: 8px;
                margin: 4px;
            }
            QLabel {
                color: #ecf0f1;
                font-size: 12px;
            }
            QSpinBox, QComboBox {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #3498db;
                border-radius: 3px;
                padding: 4px;
            }
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        
    def remove_interval(self):
        self.setParent(None)
        self.removed.emit(self)

    def set_max_frame(self, max_frame):
        self.start_frame_spinbox.setMaximum(max_frame - 1)
        self.end_frame_spinbox.setMaximum(max_frame)
        self.update_start_frame_max()
        if self.end_frame_spinbox.value() > max_frame:
            self.end_frame_spinbox.setValue(max_frame)

    def update_start_frame_max(self):
        self.start_frame_spinbox.setMaximum(self.end_frame_spinbox.value() - 1)
    
    def get_interval_data(self):
        """Returns interval data in the format expected by interpolate_intervals"""
        return {
            "start": self.start_frame_spinbox.value(),
            "stop": self.end_frame_spinbox.value(),
            "interp": int(math.log(int(self.interpolation_combo.currentText().replace("x", "")), 2))
        }


class InterpolationDialog(QDialog):
    """Dialog for configuring frame interpolation"""
    
    def __init__(self, max_frames, parent=None):
        super().__init__(parent)
        self.max_frames = max_frames
        self.setWindowTitle("Frame Interpolation (RIFE)")
        self.setMinimumWidth(600)
        
        self.layout = QVBoxLayout()
        
        # Info label
        info_label = QLabel(
            "Configure frame interpolation intervals using RIFE neural network.\n"
            "Interpolation factors: 2x, 4x, 8x, or 16x frames between each pair."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #ecf0f1; font-size: 12px; margin-bottom: 10px;")
        self.layout.addWidget(info_label)
        
        # Max frames display
        frames_label = QLabel(f"Total frames in sequence: {max_frames}")
        frames_label.setStyleSheet("color: #3498db; font-weight: bold; margin-bottom: 10px;")
        self.layout.addWidget(frames_label)
        
        # Error label
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.layout.addWidget(self.error_label)
        
        # Add interval button
        self.add_interval_button = QPushButton("Add Interval")
        self.add_interval_button.clicked.connect(self.add_interval)
        self.add_interval_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        self.layout.addWidget(self.add_interval_button)
        
        # Dialog buttons
        self.dialog_buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self
        )
        self.dialog_buttons.accepted.connect(self.check_intervals)
        self.dialog_buttons.rejected.connect(self.reject)
        
        self.layout.addWidget(self.dialog_buttons)
        
        self.setLayout(self.layout)
        
        # Track intervals
        self.intervals = []
        
        # Apply dark theme with proper text colors
        self.setStyleSheet("""
            QDialog {
                background-color: #2c3e50;
            }
            QLabel {
                color: #ecf0f1;
            }
            QDialogButtonBox QPushButton {
                background-color: #3498db;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QDialogButtonBox QPushButton:hover {
                background-color: #2980b9;
            }
        """)
    
    def add_interval(self):
        """Add a new interpolation interval widget"""
        interval_widget = InterpolationIntervalWidget(max_frame=self.max_frames)
        interval_widget.removed.connect(self.remove_interval)
        
        # Insert before the add button
        self.layout.insertWidget(self.layout.count() - 2, interval_widget)
        self.intervals.append(interval_widget)
        
    def remove_interval(self, widget):
        """Remove an interpolation interval widget"""
        if widget in self.intervals:
            self.intervals.remove(widget)
            widget.deleteLater()
    
    def remove_all_intervals(self):
        """Remove all interval widgets"""
        for widget in self.intervals[:]:
            self.remove_interval(widget)
    
    def check_intervals(self):
        """Validate intervals before accepting"""
        if not self.intervals:
            self.error_label.setText("Please add at least one interpolation interval.")
            return
        
        # Check for overlapping intervals
        interval_data = [widget.get_interval_data() for widget in self.intervals]
        
        # Sort by start frame
        interval_data.sort(key=lambda x: x['start'])
        
        # Check for overlaps
        for i in range(len(interval_data) - 1):
            if interval_data[i]['stop'] > interval_data[i+1]['start']:
                self.error_label.setText(
                    f"Intervals overlap: frames {interval_data[i]['start']}-{interval_data[i]['stop']} "
                    f"and {interval_data[i+1]['start']}-{interval_data[i+1]['stop']}"
                )
                return
        
        # Check bounds
        for interval in interval_data:
            if interval['start'] < 0 or interval['stop'] > self.max_frames:
                self.error_label.setText(
                    f"Interval {interval['start']}-{interval['stop']} is out of bounds (0-{self.max_frames})"
                )
                return
            if interval['start'] >= interval['stop']:
                self.error_label.setText(
                    f"Invalid interval: start frame ({interval['start']}) must be less than end frame ({interval['stop']})"
                )
                return
        
        self.error_label.setText("")
        self.accept()
    
    def get_intervals(self):
        """Get all configured intervals"""
        return [widget.get_interval_data() for widget in self.intervals]
