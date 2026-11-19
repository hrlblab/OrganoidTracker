"""Apply one checked, documented modification to the pinned noVNC release."""

from pathlib import Path

path = Path("/opt/novnc/core/rfb.js")
source = path.read_text()
tight = "            encs.push(encodings.encodingTight);"
zrle = "            encs.push(encodings.encodingZRLE);"
if source.count(tight) != 1 or source.count(zrle) != 1:
    raise SystemExit("Unexpected noVNC source: review the lossless-encoding patch before upgrading.")
path.write_text(source.replace(zrle + "\n", "").replace(tight, zrle + "\n" + tight))
