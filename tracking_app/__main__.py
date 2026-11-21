"""Start the desktop application."""
import os
from pathlib import Path
import subprocess
import sys
import sysconfig


def _prefer_system_tk_with_xft() -> None:
    """Use system Tk when a bundled Linux Tk lacks antialiased Xft fonts."""
    if sys.platform != "linux" or os.environ.get("TRACKING_TK_BOOTSTRAPPED"):
        return

    bundled_tk = Path(sys.base_prefix) / "lib" / "libtk8.6.so"
    multiarch = sysconfig.get_config_var("MULTIARCH")
    if not bundled_tk.is_file() or not multiarch:
        return
    system_lib = Path("/usr/lib") / multiarch
    system_tk = system_lib / "libtk8.6.so"
    system_tcl = system_lib / "libtcl8.6.so"
    if not system_tk.is_file() or not system_tcl.is_file():
        return

    try:
        bundled_links = subprocess.check_output(["ldd", str(bundled_tk)], text=True)
        system_links = subprocess.check_output(["ldd", str(system_tk)], text=True)
    except (OSError, subprocess.CalledProcessError):
        return
    if "libXft.so" in bundled_links or "libXft.so" not in system_links:
        return

    environment = os.environ.copy()
    environment["TRACKING_TK_BOOTSTRAPPED"] = "1"
    environment["LD_PRELOAD"] = " ".join(filter(None, (
        str(system_tcl), str(system_tk), environment.get("LD_PRELOAD", ""))))
    os.execve(sys.executable, sys.orig_argv, environment)


_prefer_system_tk_with_xft()

import tkinter as tk

from .ui.desktop import main

if __name__ == "__main__":
    try:
        main()
    except tk.TclError as exc:
        raise SystemExit(f"Could not open the Tkinter window: {exc}. Run this in a desktop session with an active DISPLAY.") from exc
