"""GUI styling for microinterp: color palette and reusable widget styles."""

# Color Palette
COLORS = {
    # Dark Theme
    'primary': '#3498db',      # Blue
    'primary_dark': '#2980b9',
    'primary_light': '#5dade2',
    
    'success': '#27ae60',      # Green
    'success_dark': '#229954',
    'success_light': '#52be80',
    
    'warning': '#f39c12',      # Orange
    'warning_dark': '#e67e22',
    'warning_light': '#f8c471',
    
    'danger': '#e74c3c',       # Red
    'danger_dark': '#c0392b',
    'danger_light': '#ec7063',
    
    'dark': '#2c3e50',         # Dark Blue Gray
    'dark_lighter': '#34495e',
    'dark_darker': '#1a252f',
    
    'light': '#ecf0f1',        # Light Gray
    'light_dark': '#bdc3c7',
    'light_darker': '#95a5a6',
    
    'background': '#1e1e1e',   # Very Dark (VS Code style)
    'surface': '#252526',
    'border': '#3e3e42',
    'divider': '#3e3e42',
    'focus': '#5dade2',
}

# Main Application Stylesheet
def get_main_stylesheet():
    return f"""
    QMainWindow {{
        background-color: {COLORS['background']};
    }}
    
    QWidget {{
        background-color: {COLORS['background']};
        color: {COLORS['light']};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 13px;
    }}
    
    /* Scrollbars */
    QScrollBar:vertical {{
        background: {COLORS['surface']};
        width: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:vertical {{
        background: {COLORS['dark_lighter']};
        border-radius: 6px;
        min-height: 20px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background: {COLORS['primary']};
    }}
    
    QScrollBar:horizontal {{
        background: {COLORS['surface']};
        height: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:horizontal {{
        background: {COLORS['dark_lighter']};
        border-radius: 6px;
        min-width: 20px;
    }}
    
    QScrollBar::handle:horizontal:hover {{
        background: {COLORS['primary']};
    }}
    
    QScrollBar::add-line, QScrollBar::sub-line {{
        border: none;
        background: none;
    }}
    
    /* Tooltips */
    QToolTip {{
        background-color: {COLORS['dark']};
        color: {COLORS['light']};
        border: 1px solid {COLORS['border']};
        padding: 5px;
        border-radius: 4px;
    }}
    """

# Sidebar Stylesheet
def get_sidebar_stylesheet():
    return f"""
    QWidget {{
        background-color: {COLORS['dark']};
    }}
    
    QPushButton {{
        text-align: left;
        padding: 14px 20px;
        background-color: transparent;
        border: none;
        border-left: 3px solid transparent;
        color: {COLORS['light']};
        font-size: 14px;
        font-weight: 500;
    }}
    
    QPushButton:hover {{
        background-color: {COLORS['dark_lighter']};
        border-left: 3px solid {COLORS['primary']};
    }}
    
    QPushButton:checked {{
        background-color: {COLORS['dark_lighter']};
        border-left: 3px solid {COLORS['primary']};
        color: {COLORS['primary']};
        font-weight: 600;
    }}
    
    QLabel {{
        color: {COLORS['light_darker']};
        font-size: 11px;
        padding: 12px 20px 8px 20px;
        font-weight: bold;
        letter-spacing: 0.5px;
    }}
    """

# Button Styles
def get_button_style(color='primary', hover_color=None):
    base_color = COLORS.get(color, COLORS['primary'])
    hover = hover_color or COLORS.get(f"{color}_dark", COLORS['primary_dark'])
    
    return f"""
    QPushButton {{
        background-color: {base_color};
        color: white;
        border: none;
        padding: 5px 10px;
        border-radius: 2px;
        font-weight: 500;
        font-size: 11px;
    }}
    
    QPushButton:hover {{
        background-color: {hover};
    }}
    
    QPushButton:pressed {{
        background-color: {COLORS.get(f"{color}_dark", hover)};
    }}
    
    QPushButton:disabled {{
        background-color: {COLORS['dark_lighter']};
        color: {COLORS['light_darker']};
    }}
    """

# Outline Button Style (transparent with border)
def get_outline_button_style(color='primary'):
    base_color = COLORS.get(color, COLORS['primary'])
    
    return f"""
    QPushButton {{
        background-color: transparent;
        color: {base_color};
        border: 2px solid {base_color};
        padding: 8px 18px;
        border-radius: 6px;
        font-weight: 500;
        font-size: 14px;
    }}
    
    QPushButton:hover {{
        background-color: {base_color};
        color: white;
    }}
    
    QPushButton:pressed {{
        background-color: {COLORS.get(f"{color}_dark", base_color)};
    }}
    
    QPushButton:disabled {{
        border-color: {COLORS['dark_lighter']};
        color: {COLORS['light_darker']};
    }}
    """

# GroupBox Style
def get_groupbox_style():
    return f"""
    QGroupBox {{
        background-color: transparent;
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        margin-top: 12px;
        padding: 15px;
        font-weight: 600;
        font-size: 14px;
        color: {COLORS['light']};
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 5px;
        color: {COLORS['primary']};
    }}
    """

# Input Field Styles
def get_input_style():
    return f"""
    QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {COLORS['dark_lighter']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 8px 12px;
        color: {COLORS['light']};
        font-size: 13px;
    }}
    
    QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {COLORS['primary']};
        background-color: {COLORS['dark']};
    }}
    
    QLineEdit:disabled, QTextEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
        background-color: {COLORS['surface']};
        color: {COLORS['light_darker']};
    }}
    """

# Checkbox and Radio Button Styles
def get_checkbox_style():
    return f"""
    QCheckBox, QRadioButton {{
        color: {COLORS['light']};
        spacing: 8px;
        font-size: 13px;
    }}
    
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 2px solid {COLORS['border']};
        background-color: {COLORS['dark_lighter']};
    }}
    
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background-color: {COLORS['primary']};
        border-color: {COLORS['primary']};
    }}
    
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
        border-color: {COLORS['primary']};
    }}
    
    QRadioButton::indicator {{
        border-radius: 9px;
    }}
    """

# Slider Style
def get_slider_style():
    return f"""
    QSlider::groove:horizontal {{
        background: {COLORS['dark_lighter']};
        height: 6px;
        border-radius: 3px;
    }}
    
    QSlider::handle:horizontal {{
        background: {COLORS['primary']};
        border: 2px solid {COLORS['primary_dark']};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    
    QSlider::handle:horizontal:hover {{
        background: {COLORS['primary_light']};
        border-color: {COLORS['primary']};
    }}
    
    QSlider::sub-page:horizontal {{
        background: {COLORS['primary']};
        border-radius: 3px;
    }}
    """

# Progress Bar Style
def get_progressbar_style():
    return f"""
    QProgressBar {{
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        text-align: center;
        color: {COLORS['light']};
        background-color: {COLORS['dark_lighter']};
        height: 24px;
    }}
    
    QProgressBar::chunk {{
        background-color: {COLORS['primary']};
        border-radius: 5px;
    }}
    """

# Tab Widget Style
def get_tab_style():
    return f"""
    QTabWidget::pane {{
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        background-color: {COLORS['surface']};
    }}
    
    QTabBar::tab {{
        background-color: {COLORS['dark']};
        color: {COLORS['light_darker']};
        padding: 10px 20px;
        border: 1px solid {COLORS['border']};
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        margin-right: 2px;
    }}
    
    QTabBar::tab:selected {{
        background-color: {COLORS['surface']};
        color: {COLORS['primary']};
        border-bottom: 2px solid {COLORS['primary']};
    }}
    
    QTabBar::tab:hover:!selected {{
        background-color: {COLORS['dark_lighter']};
        color: {COLORS['light']};
    }}
    """

# Label Styles
def get_label_style(size='normal', weight='normal', color='light'):
    sizes = {'small': '11px', 'normal': '13px', 'large': '16px', 'xlarge': '20px', 'title': '24px'}
    weights = {'normal': 'normal', 'medium': '500', 'semibold': '600', 'bold': 'bold'}
    
    return f"""
    QLabel {{
        color: {COLORS.get(color, COLORS['light'])};
        font-size: {sizes.get(size, '13px')};
        font-weight: {weights.get(weight, 'normal')};
    }}
    """

# Card/Panel Style
def get_card_style():
    return f"""
    QFrame {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 15px;
    }}
    """

# Menu Style
def get_menu_style():
    return f"""
    QMenu {{
        background-color: {COLORS['dark']};
        border: 1px solid {COLORS['border']};
        padding: 5px;
    }}
    
    QMenu::item {{
        padding: 8px 25px 8px 15px;
        color: {COLORS['light']};
        border-radius: 4px;
    }}
    
    QMenu::item:selected {{
        background-color: {COLORS['primary']};
    }}
    
    QMenu::separator {{
        height: 1px;
        background: {COLORS['border']};
        margin: 5px 10px;
    }}
    """

# Combined stylesheet getter
def get_combined_styles():
    """Returns all styles combined"""
    return (
        get_main_stylesheet() + 
        get_input_style() + 
        get_checkbox_style() + 
        get_slider_style() + 
        get_progressbar_style() + 
        get_tab_style() + 
        get_menu_style()
    )
