"""Explicit acquisition selection on the Tk thread, before predictor initialization."""

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from ..core.video_frames import selected_frame_indices
from ..services.session import Timing


def choose_frames(parent, frames, span):
    dialog = tk.Toplevel(parent)
    dialog.title("Confirm acquisition time points")
    dialog.transient(parent)
    preview = ttk.Label(dialog)
    preview.pack()
    source_label = ttk.Label(dialog)
    source_label.pack()

    def show(value):
        index = int(float(value))
        image = Image.fromarray(frames[index])
        image.thumbnail((720, 420))
        preview.image = ImageTk.PhotoImage(image, master=dialog)
        preview.configure(image=preview.image)
        source_label.configure(text=f"Source frame {index + 1} / {len(frames)}")

    tk.Scale(dialog, from_=0, to=len(frames) - 1, orient="horizontal", command=show).pack(fill="x")
    show(0)
    indices = tk.StringVar(dialog, ", ".join(str(i + 1) for i in range(len(frames))))
    times = tk.StringVar(
        dialog, ", ".join(f"{1 + i * span / max(len(frames) - 1, 1):.12g}" for i in range(len(frames)))
    )
    ttk.Label(dialog, text="Source frame numbers (starting at 1, in chronological order)").pack()
    ttk.Entry(dialog, textvariable=indices, width=65).pack(fill="x")
    ttk.Label(dialog, text="Acquisition days (one day value per selected source frame)").pack()
    ttk.Entry(dialog, textvariable=times, width=65).pack(fill="x")

    def preset():
        indices.set("1, 3, 4, 6, 7, 9, 10")
        times.set("1, 2, 3, 4, 5, 6, 7")

    if len(frames) == 11:
        ttk.Button(dialog, text="7 daily points (confirmed 11-frame export only)", command=preset).pack()
    status = ttk.Label(dialog, text="Inspect the source frames before confirming. Image similarity is not a time axis.")
    status.pack()
    result = None

    def confirm():
        nonlocal result
        try:
            keep = selected_frame_indices(len(frames), [int(v.strip()) - 1 for v in indices.get().split(",")])
            timing = Timing(frame_times_days=tuple(float(v.strip()) for v in times.get().split(",")))
            timing.resolve(len(keep))
        except ValueError as error:
            status.configure(text=str(error))
            return
        result = (tuple(keep), timing.frame_times_days)
        dialog.destroy()

    ttk.Button(dialog, text="Confirm frames and days", command=confirm).pack()
    ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack()
    dialog.grab_set()
    parent.wait_window(dialog)
    return result
