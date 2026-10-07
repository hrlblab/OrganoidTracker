#!/usr/bin/env python3
"""
Progress Dialog Widget
Shows progress during long-running operations like tracking
"""

import tkinter as tk
from tkinter import ttk


class ProgressDialog:
    """
    Modal progress dialog for showing operation progress
    """

    def __init__(self, parent, title="Processing", message="Please wait..."):
        """
        Initialize the progress dialog

        Args:
            parent: Parent window
            title: Dialog title
            message: Initial message
        """
        self.parent = parent
        self.dialog = None
        self.progress_var = tk.DoubleVar()
        self.message_var = tk.StringVar(value=message)
        self.title = title
        self.progress_text = None  # Initialize to None, will be set in create_widgets()
        
        # Throttling mechanism for long videos
        self.last_update_time = 0
        self.min_update_interval = 0.1  # Minimum 100ms between updates

    def show(self):
        """Show the progress dialog"""
        # Create dialog window
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title(self.title)
        self.dialog.geometry("400x150")
        self.dialog.resizable(False, False)

        # Center on parent
        self.dialog.transient(self.parent)

        # Create widgets first
        self.create_widgets()

        # Center the dialog
        self.center_dialog()

        # Update the dialog to ensure it's fully rendered
        self.dialog.update_idletasks()

        # Try to grab focus, but handle errors gracefully
        try:
            self.dialog.grab_set()
        except tk.TclError as e:
            print(f"⚠️ Warning: Could not grab dialog focus: {e}")
            # Continue without modal behavior

        # Start with 0 progress
        self.progress_var.set(0)

    def center_dialog(self):
        """Center dialog on parent window"""
        # Update both parent and dialog to get accurate dimensions
        self.parent.update_idletasks()
        self.dialog.update_idletasks()

        # Get parent window position and size
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        # Get dialog size (with fallback to requested size)
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()

        # If dialog hasn't been rendered yet, use requested size
        if dialog_width <= 1:
            dialog_width = self.dialog.winfo_reqwidth()
        if dialog_height <= 1:
            dialog_height = self.dialog.winfo_reqheight()

        # Ensure minimum dialog size
        dialog_width = max(400, dialog_width)
        dialog_height = max(150, dialog_height)

        # Calculate center position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2

        # Ensure dialog doesn't go off-screen
        x = max(0, x)
        y = max(0, y)

        # Apply geometry with centering
        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")

        # Force the dialog to be on top and centered
        self.dialog.lift()
        self.dialog.focus_force()

    def create_widgets(self):
        """Create dialog widgets"""
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.grid(row=0, column=0, sticky='ew')

        # Message label
        message_label = ttk.Label(
            main_frame,
            textvariable=self.message_var,
            font=('Arial', 10)
        )
        message_label.grid(row=0, column=0, pady=(0, 15), sticky='ew')

        # Progress bar
        progress_bar = ttk.Progressbar(
            main_frame,
            variable=self.progress_var,
            mode='determinate',
            length=300
        )
        progress_bar.grid(row=1, column=0, pady=(0, 10), sticky='ew')

        # Progress text
        self.progress_text = ttk.Label(
            main_frame,
            text="0%",
            font=('Arial', 9)
        )
        self.progress_text.grid(row=2, column=0, sticky='ew')

        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        self.dialog.columnconfigure(0, weight=1)

    def update_progress(self, progress, message=""):
        """
        Update progress and message with throttling for long videos

        Args:
            progress: Progress value (0-100)
            message: Progress message
        """
        if self.dialog is None:
            return

        # Throttling: skip updates that are too frequent (except for completion)
        import time
        current_time = time.time()
        if progress < 100 and (current_time - self.last_update_time) < self.min_update_interval:
            return
        self.last_update_time = current_time

        # Ensure progress is valid
        progress = max(0, min(100, progress))

        try:
            self.progress_var.set(progress)

            if message:
                self.message_var.set(message)

            # Update progress text only if the widget exists
            if hasattr(self, 'progress_text') and self.progress_text:
                try:
                    self.progress_text.config(text=f"{progress:.1f}%")
                except tk.TclError:
                    # Widget might have been destroyed
                    pass

            # Force immediate update of the dialog (use update_idletasks to prevent recursion)
            if self.dialog and hasattr(self.dialog, 'update_idletasks'):
                self.dialog.update_idletasks()

        except tk.TclError:
            # Dialog might have been destroyed
            return

        # Auto-close when complete (with a longer delay for user to see completion)
        if progress >= 100:
            # Check if dialog still exists before scheduling auto-close
            if self.dialog and hasattr(self.dialog, 'after'):
                try:
                    self.dialog.after(2000, self.close)  # Close after 2 seconds to show completion
                except tk.TclError:
                    # Dialog might have been destroyed
                    pass

    def close(self):
        """Close the dialog"""
        if self.dialog:
            try:
                self.dialog.grab_release()
            except tk.TclError:
                # Grab might not have been set or dialog already destroyed
                pass

            try:
                self.dialog.destroy()
            except tk.TclError:
                # Dialog might already be destroyed
                pass

            self.dialog = None

    def set_message(self, message):
        """Set the message text"""
        self.message_var.set(message)