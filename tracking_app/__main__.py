"""Start the desktop application."""
import tkinter as tk

from .ui.desktop import main

if __name__ == "__main__":
    try:
        main()
    except tk.TclError as exc:
        raise SystemExit(f"Could not open the Tkinter window: {exc}. Run this in a desktop session with an active DISPLAY.") from exc
