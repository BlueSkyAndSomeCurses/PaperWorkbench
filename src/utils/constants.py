SUPPORTED_TABLE_FORMATS = [".csv", ".parquet"]

CAPTURE_CODE = """
# --- kiroku: capture figures to base64 stdout ---
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as _plt
    import io as _io, base64 as _b64
    
    # Ensure we have at least one figure
    figs = _plt.get_fignums()
    if not figs:
        print("No figures found, creating default figure")
        _plt.figure(figsize=(8, 6))
        _plt.text(0.5, 0.5, 'Plot generation failed', ha='center', va='center')
        _plt.title('Error: No plot generated')
        figs = _plt.get_fignums()
    
    print(f"Found {len(figs)} figure(s)")
    for _i in figs:
        _fig = _plt.figure(_i)
        _buf = _io.BytesIO()
        _fig.savefig(_buf, format='png', bbox_inches='tight', dpi=100)
        _buf.seek(0)
        _enc = _b64.b64encode(_buf.getvalue()).decode('ascii')
        print('data:image/png;base64,' + _enc)
        print(f"Generated image for figure {_i}")
        
except Exception as _e:
    print(f'capture_error: {_e}')
    import traceback
    print(f'capture_traceback: {traceback.format_exc()}')
"""

MATPLOTLIB_SETUP = """
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
"""

DATA_BOOTSTRAP = """
import pandas as pd
data = pd.read_csv('data.csv')
"""