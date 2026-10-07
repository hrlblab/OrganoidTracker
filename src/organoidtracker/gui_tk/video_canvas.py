#!/usr/bin/env python3
"""
Video Canvas Widget
Custom Tkinter canvas for displaying video frames and handling click interactions
"""

import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
import numpy as np
from typing import Callable, Optional, Tuple


class VideoCanvas:
    """
    Custom canvas widget for video display and interaction

    Features:
    - Display video frames
    - Handle mouse clicks for prompting
    - Visual feedback for click positions
    - Proper scaling and aspect ratio handling
    """

    def __init__(self, parent, width=600, height=400, **kwargs):
        """Initialize the video canvas"""
        self.parent = parent
        self.canvas = Canvas(parent, width=width, height=height, bg='black', **kwargs)

        # Video display properties
        self.current_image_tk = None
        self.original_frame = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Bounding box markers - now organized by object ID
        self.markers = {}  # {obj_id: [list of bounding box marker IDs]}
        self.marker_counter = 0

        # Bounding box creation state
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_current_x = 0
        self.drag_current_y = 0
        self.preview_bbox_id = None  # ID of the preview rectangle during dragging

        # Object colors - expanded palette for 20+ objects
        self.object_colors = {
            0: '#808080',   # Gray - Background
            1: '#FF0000',   # Red
            2: '#00FF00',   # Green
            3: '#0000FF',   # Blue
            4: '#FFFF00',   # Yellow
            5: '#FF00FF',   # Magenta
            6: '#00FFFF',   # Cyan
            7: '#FFA500',   # Orange
            8: '#800080',   # Purple
            9: '#FFC0CB',   # Pink
            10: '#A52A2A',  # Brown
            11: '#90EE90',  # Light Green
            12: '#87CEEB',  # Sky Blue
            13: '#DDA0DD',  # Plum
            14: '#F0E68C',  # Khaki
            15: '#FF6347',  # Tomato
            16: '#40E0D0',  # Turquoise
            17: '#EE82EE',  # Violet
            18: '#FFB6C1',  # Light Pink
            19: '#98FB98',  # Pale Green
            20: '#F5DEB3',  # Wheat
        }

        # Callback functions for different interactions
        self.bbox_callback = None
        self.click_callback = None

        # Bind events for bounding box creation
        self.canvas.bind("<Button-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        # Show placeholder
        self.show_placeholder()

    def grid(self, **kwargs):
        """Grid the canvas widget"""
        self.canvas.grid(**kwargs)

    def pack(self, **kwargs):
        """Pack the canvas"""
        self.canvas.pack(**kwargs)

    def show_placeholder(self):
        """Show placeholder text when no video is loaded"""
        self.canvas.delete("all")

        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Fallback to configured dimensions if not yet rendered
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = int(self.canvas['width'])
            canvas_height = int(self.canvas['height'])

        text = "Load a video to see the first frame here\nClick for organoids, drag for cyst bounding boxes\nNumbers will show object identities"

        self.canvas.create_text(
            canvas_width // 2,
            canvas_height // 2,
            text=text,
            fill='white',
            font=('Arial', 12),
            justify='center'
        )

    def display_frame(self, frame: np.ndarray):
        """
        Display a video frame on the canvas

        Args:
            frame: Video frame as numpy array (RGB format)
        """
        try:
            self.original_frame = frame.copy()

            # Convert to PIL Image
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            pil_image = Image.fromarray(frame)

            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # If canvas not yet rendered, use configured dimensions
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = int(self.canvas['width'])
                canvas_height = int(self.canvas['height'])

            # Calculate scaling to fit canvas while maintaining aspect ratio
            img_width, img_height = pil_image.size
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            self.scale_factor = min(scale_x, scale_y)

            # Resize image
            new_width = int(img_width * self.scale_factor)
            new_height = int(img_height * self.scale_factor)

            resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Calculate centering offset
            self.offset_x = (canvas_width - new_width) // 2
            self.offset_y = (canvas_height - new_height) // 2

            # Convert to PhotoImage and display
            self.current_image_tk = ImageTk.PhotoImage(resized_image)

            self.canvas.delete("all")
            self.canvas.create_image(
                self.offset_x,
                self.offset_y,
                anchor='nw',
                image=self.current_image_tk
            )

            # Redraw click markers
            self.redraw_markers()

        except Exception as e:
            print(f"Error displaying frame: {e}")
            self.show_error_message(f"Error displaying frame: {e}")

    def show_error_message(self, message: str):
        """Show error message on canvas"""
        self.canvas.delete("all")

        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Fallback to configured dimensions if not yet rendered
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = int(self.canvas['width'])
            canvas_height = int(self.canvas['height'])

        self.canvas.create_text(
            canvas_width // 2,
            canvas_height // 2,
            text=f"Error: {message}",
            fill='red',
            font=('Arial', 10),
            justify='center'
        )

    def canvas_to_image_coords(self, canvas_x: int, canvas_y: int) -> Optional[Tuple[int, int]]:
        """
        Convert canvas coordinates to original image coordinates

        Args:
            canvas_x, canvas_y: Coordinates on canvas

        Returns:
            Tuple of (x, y) in original image coordinates, or None if outside image
        """
        if self.original_frame is None:
            return None

        # Adjust for image offset
        img_x = canvas_x - self.offset_x
        img_y = canvas_y - self.offset_y

        # Check if click is within image bounds
        img_height, img_width = self.original_frame.shape[:2]
        scaled_width = int(img_width * self.scale_factor)
        scaled_height = int(img_height * self.scale_factor)

        if img_x < 0 or img_y < 0 or img_x >= scaled_width or img_y >= scaled_height:
            return None

        # Scale back to original image coordinates
        orig_x = int(img_x / self.scale_factor)
        orig_y = int(img_y / self.scale_factor)

        # Clamp to image bounds
        orig_x = max(0, min(orig_x, img_width - 1))
        orig_y = max(0, min(orig_y, img_height - 1))

        return (orig_x, orig_y)

    def image_to_canvas_coords(self, img_x: int, img_y: int) -> Tuple[int, int]:
        """
        Convert original image coordinates to canvas coordinates

        Args:
            img_x, img_y: Coordinates in original image

        Returns:
            Tuple of (x, y) in canvas coordinates
        """
        canvas_x = int(img_x * self.scale_factor) + self.offset_x
        canvas_y = int(img_y * self.scale_factor) + self.offset_y

        return (canvas_x, canvas_y)

    def on_mouse_press(self, event):
        """Handle mouse button press - either click or start drag"""
        if self.original_frame is None:
            return

        # Convert to image coordinates
        image_coords = self.canvas_to_image_coords(event.x, event.y)

        if image_coords is None:
            return  # Click was outside image

        # Store initial position for both click and drag detection
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.drag_current_x = event.x
        self.drag_current_y = event.y

        # We'll determine if this is a click or drag in mouse_release
        self.dragging = False  # Will be set to True in mouse_drag if movement detected

        # Clear any existing preview
        if self.preview_bbox_id:
            self.canvas.delete(self.preview_bbox_id)
            self.preview_bbox_id = None

    def on_mouse_drag(self, event):
        """Handle mouse drag - update preview bounding box"""
        if self.original_frame is None:
            return

        # Calculate movement from start position
        dx = abs(event.x - self.drag_start_x)
        dy = abs(event.y - self.drag_start_y)

        # Only start dragging if moved more than a few pixels (prevents accidental drags)
        if dx > 3 or dy > 3:
            self.dragging = True

        if not self.dragging:
            return

        # Update current position
        self.drag_current_x = event.x
        self.drag_current_y = event.y

        # Clear previous preview
        if self.preview_bbox_id:
            self.canvas.delete(self.preview_bbox_id)

        # Create preview rectangle
        self.preview_bbox_id = self.canvas.create_rectangle(
            self.drag_start_x, self.drag_start_y,
            self.drag_current_x, self.drag_current_y,
            outline='white', width=2, dash=(5, 5)  # Dashed white outline for preview
        )

    def on_mouse_release(self, event):
        """Handle mouse button release - either click or finalize bounding box"""
        if self.original_frame is None:
            return

        # Clear preview
        if self.preview_bbox_id:
            self.canvas.delete(self.preview_bbox_id)
            self.preview_bbox_id = None

        if self.dragging:
            # This was a drag operation - create bounding box
            self.dragging = False

            # Get final coordinates
            start_coords = self.canvas_to_image_coords(self.drag_start_x, self.drag_start_y)
            end_coords = self.canvas_to_image_coords(event.x, event.y)

            if start_coords is None or end_coords is None:
                return  # Outside image

            # Ensure proper bounding box format (x1 <= x2, y1 <= y2)
            x1, y1 = start_coords
            x2, y2 = end_coords

            # Normalize coordinates
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Minimum bounding box size (10x10 pixels)
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                return  # Too small to be useful

            # Call the bbox callback if set
            if self.bbox_callback:
                self.bbox_callback(x1, y1, x2, y2)
        else:
            # This was a click operation (no significant drag)
            click_coords = self.canvas_to_image_coords(event.x, event.y)

            if click_coords is None:
                return  # Click was outside image

            # Call the click callback if set
            if self.click_callback:
                x, y = click_coords
                self.click_callback(x, y)

    def on_mouse_move(self, event):
        """Handle mouse movement (currently used for dragging preview)"""
        # Future: Could show hover effects or cursor changes
        pass

    def set_bbox_callback(self, callback: Callable[[int, int, int, int], None]):
        """
        Set callback function for bounding box events

        Args:
            callback: Function to call on bbox creation, signature: (x1, y1, x2, y2)
        """
        self.bbox_callback = callback

    def set_click_callback(self, callback: Callable[[int, int], None]):
        """
        Set callback function for click events

        Args:
            callback: Function to call on click, signature: (x, y)
        """
        self.click_callback = callback

    def add_bbox_marker(self, img_x1: int, img_y1: int, img_x2: int, img_y2: int, obj_id: int = 1):
        """Add a bounding box marker to the canvas (box only, no labels)"""
        # Convert image coordinates to canvas coordinates
        canvas_x1, canvas_y1 = self.image_to_canvas_coords(img_x1, img_y1)
        canvas_x2, canvas_y2 = self.image_to_canvas_coords(img_x2, img_y2)

        # Get color for this object
        color = self.object_colors.get(obj_id, '#FF0000')

        # Create bounding box rectangle
        bbox_id = self.canvas.create_rectangle(
            canvas_x1, canvas_y1, canvas_x2, canvas_y2,
            outline=color, width=2, fill='', stipple='gray25'  # Semi-transparent fill
        )

        # Store marker components
        if obj_id not in self.markers:
            self.markers[obj_id] = []

        # Store just the bounding box ID
        self.markers[obj_id].append(bbox_id)

    def add_organoid_marker(self, img_x: int, img_y: int, organoid_id: int):
        """Add an organoid identification marker (small number only, no circle)"""
        # Convert image coordinates to canvas coordinates
        canvas_x, canvas_y = self.image_to_canvas_coords(img_x, img_y)

        # Add text label only - small, clear, noticeable color
        text_id = self.canvas.create_text(
            canvas_x, canvas_y,
            text=str(organoid_id),  # Just the number
            fill='#00FF00',  # Bright green for high visibility
            font=('Arial', 8, 'bold'),  # Small, bold font (half size)
            anchor='center'
        )

        # Store organoid markers separately (no circle)
        marker_key = f"organoid_{organoid_id}"
        if marker_key not in self.markers:
            self.markers[marker_key] = []

        self.markers[marker_key].append(text_id)

    def clear_markers(self, obj_id: Optional[int] = None):
        """Clear bounding box markers for specific object or all markers"""
        if obj_id is not None:
            # Clear markers for specific object
            if obj_id in self.markers:
                for bbox_id in self.markers[obj_id]:
                    # marker is just the bounding box ID
                    self.canvas.delete(bbox_id)     # Delete the bounding box
                del self.markers[obj_id]
        else:
            # Clear all markers (including organoid markers)
            for obj_markers in self.markers.values():
                for marker_id in obj_markers:
                    # marker could be bbox, circle, or text
                    self.canvas.delete(marker_id)
            self.markers.clear()

    def get_marker_count(self, obj_id: Optional[int] = None) -> int:
        """Get number of markers for specific object or total"""
        if obj_id is not None:
            return len(self.markers.get(obj_id, []))
        else:
            return sum(len(obj_markers) for obj_markers in self.markers.values())

    def redraw_markers(self):
        """Redraw markers after image update (placeholder for now)"""
        # For now, markers are automatically redrawn when needed
        # The canvas.delete("all") removes them, but they can be re-added by the GUI
        pass

    def get_canvas_size(self) -> Tuple[int, int]:
        """Get canvas dimensions"""
        return (int(self.canvas['width']), int(self.canvas['height']))

    def update_size(self, width: int, height: int):
        """Update canvas size"""
        self.canvas.config(width=width, height=height)

        # Redisplay current frame if available
        if self.original_frame is not None:
            self.display_frame(self.original_frame)