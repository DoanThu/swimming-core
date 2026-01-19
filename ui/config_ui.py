# Coach View UI Configuration
UI_FPS = 30
CHART_UPDATE_INTERVAL = 1 # seconds
FRAMES_PER_UPDATE = UI_FPS * CHART_UPDATE_INTERVAL
MAX_HISTORY_FRAMES = UI_FPS * 10 # Keep 10 seconds of history
TIMEOUT = 5.0 # seconds before considering a swimmer inactive
SMOOTH_WINDOW = 10
NUM_CHARTS = 3
VIDEO_TARGET_HEIGHT = 550
MIN_SPEED = 0.5 # m/s
MAX_SPEED = 2.5 # m/s
DATA_FOLDER = "ui/data"
DATA_CSV_FILE = "swimming_session_data.csv"

# CSV Column Names
CSV_COLUMNS = [
    "Date",           # Date of the session
    "Time",           # Time of the session
    "SessionID",      # Session identifier
    "Lane",           # Lane number
    "AvgSpeed",       # Average speed in m/s
    "AvgStrokeRate",  # Average stroke rate in SPM (strokes per minute)
    "AvgDPS"          # Average distance per stroke in meters
]


# Athlete View UI Configuration
CELL_FONT_SIZE = '4.0rem'
HEADER_FONT_SIZE = '3.0rem'