#!/usr/bin/env python3
"""
Results Viewer Window
Display tracking results frame by frame
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Dict, List, Any
from .video_canvas import VideoCanvas


class ResultsViewer:
"""
Window for viewing tracking results frame by frame

Features:
- Frame navigation
- Original vs segmented view
- Frame-by-frame analysis
"""

def __init__(self, parent, frames: List[np.ndarray], video_segments: Dict[int, Dict[int, Any]], obj_id: int = 1):
"""
Initialize the results viewer

Args:
parent: Parent window
frames: List of original video frames
video_segments: Tracking results
obj_id: Object ID to display
"""
self.parent = parent
self.frames = frames
self.video_segments = video_segments
self.obj_id = obj_id

self.current_frame_idx = 0
self.view_mode = "overlay" # "original", "overlay", "mask", "side_by_side"

# Create window
self.window = tk.Toplevel(parent)
self.window.title("Tracking Results Viewer")
self.window.geometry("800x700")

self.setup_widgets()
self.setup_layout()
self.update_display()

def setup_widgets(self):
"""Create all widgets"""
# Title
self.title_label = ttk.Label(
self.window,
text="Tracking Results Viewer",
font=('Arial', 14, 'bold')
)

# Controls frame
self.controls_frame = ttk.Frame(self.window)

# Navigation controls
self.nav_frame = ttk.LabelFrame(self.controls_frame, text="Navigation", padding=5)

self.prev_btn = ttk.Button(
self.nav_frame,
text="◀ Previous",
command=self.prev_frame
)

self.frame_var = tk.StringVar(value="Frame 1 / 1")
self.frame_label = ttk.Label(
self.nav_frame,
textvariable=self.frame_var,
font=('Arial', 10, 'bold')
)

self.next_btn = ttk.Button(
self.nav_frame,
text="Next ▶",
command=self.next_frame
)

# Frame slider
self.frame_scale = ttk.Scale(
self.nav_frame,
from_=0,
to=len(self.frames) - 1,
orient='horizontal',
command=self.on_scale_change
)

# View mode controls
self.view_frame = ttk.LabelFrame(self.controls_frame, text="View Mode", padding=5)

self.view_var = tk.StringVar(value=self.view_mode)

self.original_radio = ttk.Radiobutton(
self.view_frame,
text="Original",
variable=self.view_var,
value="original",
command=self.on_view_mode_change
)

self.overlay_radio = ttk.Radiobutton(
self.view_frame,
text="Overlay",
variable=self.view_var,
value="overlay",
command=self.on_view_mode_change
)

self.mask_radio = ttk.Radiobutton(
self.view_frame,
text="Mask",
variable=self.view_var,
value="mask",
command=self.on_view_mode_change
)

self.side_radio = ttk.Radiobutton(
self.view_frame,
text="Side-by-Side",
variable=self.view_var,
value="side_by_side",
command=self.on_view_mode_change
)

# Display canvas
self.canvas = VideoCanvas(self.window, width=700, height=400)

# Info frame
self.info_frame = ttk.LabelFrame(self.window, text="Frame Information", padding=5)

self.info_text = tk.Text(
self.info_frame,
height=4,
width=50,
state='disabled',
font=('Courier', 9)
)

# Close button
self.close_btn = ttk.Button(
self.window,
text="Close",
command=self.close
)

def setup_layout(self):
"""Setup widget layout"""
# Title
self.title_label.grid(row=0, column=0, pady=10, sticky='ew')

# Controls
self.controls_frame.grid(row=1, column=0, padx=10, pady=5, sticky='ew')

# Navigation frame layout
self.nav_frame.grid(row=0, column=0, padx=5, sticky='ew')

self.prev_btn.grid(row=0, column=0, padx=5, pady=2)
self.frame_label.grid(row=0, column=1, padx=10, pady=2)
self.next_btn.grid(row=0, column=2, padx=5, pady=2)

self.frame_scale.grid(row=1, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

self.nav_frame.columnconfigure(1, weight=1)

# View mode frame layout
self.view_frame.grid(row=0, column=1, padx=5, sticky='ew')

self.original_radio.grid(row=0, column=0, padx=3, pady=2)
self.overlay_radio.grid(row=0, column=1, padx=3, pady=2)
self.mask_radio.grid(row=1, column=0, padx=3, pady=2)
self.side_radio.grid(row=1, column=1, padx=3, pady=2)

# Configure controls frame
self.controls_frame.columnconfigure(0, weight=1)

# Canvas
self.canvas.grid(row=2, column=0, padx=10, pady=5)

# Info frame
self.info_frame.grid(row=3, column=0, padx=10, pady=5, sticky='ew')
self.info_text.grid(row=0, column=0, sticky='ew')
self.info_frame.columnconfigure(0, weight=1)

# Close button
self.close_btn.grid(row=4, column=0, pady=10)

# Configure main window
self.window.columnconfigure(0, weight=1)

def update_display(self):
"""Update the display with current frame and view mode"""
if not self.frames:
return

frame_idx = self.current_frame_idx
frame = self.frames[frame_idx].copy()

# Update frame counter
self.frame_var.set(f"Frame {frame_idx + 1} / {len(self.frames)}")
self.frame_scale.set(frame_idx)

# Generate display frame based on view mode
display_frame = self.generate_display_frame(frame, frame_idx)

# Show frame
self.canvas.display_frame(display_frame)

# Update info
self.update_info(frame_idx)

# Update navigation buttons
self.prev_btn.config(state='normal' if frame_idx > 0 else 'disabled')
self.next_btn.config(state='normal' if frame_idx < len(self.frames) - 1 else 'disabled')

def generate_display_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
"""
Generate display frame based on view mode

Args:
frame: Original frame
frame_idx: Frame index

Returns:
Display frame
"""
if self.view_mode == "original":
return frame

elif self.view_mode == "overlay":
return self.create_overlay_frame(frame, frame_idx)

elif self.view_mode == "mask":
return self.create_mask_frame(frame, frame_idx)

elif self.view_mode == "side_by_side":
return self.create_side_by_side_frame(frame, frame_idx)

return frame

def create_overlay_frame(self, frame: np.ndarray, frame_idx: int, alpha: float = 0.3) -> np.ndarray:
"""Create frame with segmentation overlay"""
overlay_frame = frame.copy()

if frame_idx in self.video_segments and self.obj_id in self.video_segments[frame_idx]:
mask = self.video_segments[frame_idx][self.obj_id].cpu().numpy()
if mask.ndim > 2:
mask = mask.squeeze()

mask_binary = mask > 0.5

# Resize mask to match frame if needed
if mask_binary.shape != frame.shape[:2]:
import cv2
mask_binary = cv2.resize(
mask_binary.astype(np.uint8),
(frame.shape[1], frame.shape[0]),
interpolation=cv2.INTER_NEAREST
).astype(bool)

# Apply red overlay
overlay_frame[mask_binary] = overlay_frame[mask_binary] * (1 - alpha) + np.array([255, 0, 0]) * alpha

return overlay_frame.astype(np.uint8)

def create_mask_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
"""Create mask-only frame"""
if frame_idx in self.video_segments and self.obj_id in self.video_segments[frame_idx]:
mask = self.video_segments[frame_idx][self.obj_id].cpu().numpy()
if mask.ndim > 2:
mask = mask.squeeze()

mask_binary = (mask > 0.5).astype(np.uint8)

# Resize mask to match frame if needed
if mask_binary.shape != frame.shape[:2]:
import cv2
mask_binary = cv2.resize(
mask_binary,
(frame.shape[1], frame.shape[0]),
interpolation=cv2.INTER_NEAREST
)

# Convert to 3-channel image (white mask on black background)
mask_frame = np.stack([mask_binary * 255] * 3, axis=-1)
return mask_frame.astype(np.uint8)
else:
# Return black frame if no mask
return np.zeros_like(frame)

def create_side_by_side_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
"""Create side-by-side comparison"""
original = frame.copy()
overlay = self.create_overlay_frame(frame, frame_idx)

# Concatenate horizontally
side_by_side = np.hstack([original, overlay])
return side_by_side

def update_info(self, frame_idx: int):
"""Update frame information"""
info_text = f"Frame {frame_idx + 1} of {len(self.frames)}\n"
info_text += f"Object ID: {self.obj_id}\n"

if frame_idx in self.video_segments and self.obj_id in self.video_segments[frame_idx]:
mask = self.video_segments[frame_idx][self.obj_id].cpu().numpy()
if mask.ndim > 2:
mask = mask.squeeze()

mask_binary = mask > 0.5
mask_area = np.sum(mask_binary)
total_pixels = mask_binary.size
coverage = (mask_area / total_pixels) * 100 if total_pixels > 0 else 0

info_text += f"Segmentation: Found\n"
info_text += f"Coverage: {coverage:.1f}% ({mask_area} pixels)"
else:
info_text += f"Segmentation: Not found\n"
info_text += f"Coverage: 0.0%"

self.info_text.config(state='normal')
self.info_text.delete(1.0, tk.END)
self.info_text.insert(1.0, info_text)
self.info_text.config(state='disabled')

def prev_frame(self):
"""Go to previous frame"""
if self.current_frame_idx > 0:
self.current_frame_idx -= 1
self.update_display()

def next_frame(self):
"""Go to next frame"""
if self.current_frame_idx < len(self.frames) - 1:
self.current_frame_idx += 1
self.update_display()

def on_scale_change(self, value):
"""Handle frame slider change"""
try:
new_frame_idx = int(float(value))
if 0 <= new_frame_idx < len(self.frames):
self.current_frame_idx = new_frame_idx
self.update_display()
except ValueError:
pass

def on_view_mode_change(self):
"""Handle view mode change"""
self.view_mode = self.view_var.get()
self.update_display()

def show(self):
"""Show the results viewer window"""
self.window.transient(self.parent)
self.window.focus_set()

def close(self):
"""Close the viewer window"""
self.window.destroy()