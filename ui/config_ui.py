# Constants for UI rendering
UI_FPS = 30
CHART_UPDATE_INTERVAL = 1 # seconds
FRAMES_PER_UPDATE = UI_FPS * CHART_UPDATE_INTERVAL
MAX_HISTORY_FRAMES = UI_FPS * 10 # Keep 10 seconds of history
TIMEOUT = 5.0 # seconds before considering a swimmer inactive
SMOOTH_WINDOW = 10
NUM_CHARTS = 3

VIDEO_TARGET_HEIGHT = 300