"""Select matplotlib's Agg backend for the analysis exports.

The analysis only writes figures to files and runs in a worker thread, so it must never open a
window or create Tk interpreters. Without this, matplotlib picks an interactive backend wherever
a display exists; on Windows that is the Tk backend, which creates one Tk interpreter per figure
and fails ("Can't find a usable init.tcl") after a few figures with Python 3.12 and 3.13.
Imported first by ``organoidtracker.analysis`` so that it precedes every ``pyplot`` import.
Embedded canvases in a GUI (QtAgg) are created explicitly and are not affected.
"""

import matplotlib

matplotlib.use("Agg")
