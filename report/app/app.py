import argparse
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Default target directory is the report folder
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent.parent.resolve()

app = FastAPI(title="Local Overleaf Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPORT_DIR = DEFAULT_REPORT_DIR


def set_report_dir(path: Path):
    global REPORT_DIR
    REPORT_DIR = path.resolve()


class SaveRequest(BaseModel):
    path: str
    content: str


class CreateFileRequest(BaseModel):
    path: str
    content: Optional[str] = ""


class DeleteFileRequest(BaseModel):
    path: str


class CompileRequest(BaseModel):
    passes: Optional[int] = 1
    engine: Optional[str] = "pdflatex"
    main_file: Optional[str] = "main.tex"


@app.get("/api/project")
def get_project_info():
    main_pdf = REPORT_DIR / "main.pdf"
    has_pdf = main_pdf.exists()
    pdf_size = main_pdf.stat().st_size if has_pdf else 0
    pdf_mtime = main_pdf.stat().st_mtime if has_pdf else 0

    return {
        "project_name": REPORT_DIR.name,
        "directory": str(REPORT_DIR),
        "has_pdf": has_pdf,
        "pdf_size": pdf_size,
        "pdf_mtime": pdf_mtime,
        "main_file": "main.tex",
    }


@app.get("/api/files")
def list_files():
    """List all relevant files in the report directory."""
    items = []
    ignored_exts = {".aux", ".toc", ".out", ".log", ".nav", ".snm", ".vrb", ".synctex.gz", ".fdb_latexmk", ".fls"}

    for root, dirs, files in os.walk(REPORT_DIR):
        # Skip hidden directories and the app directory itself
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "app"]

        rel_root = os.path.relpath(root, REPORT_DIR)
        if rel_root == ".":
            rel_root = ""

        for d in dirs:
            dir_rel = os.path.join(rel_root, d)
            items.append({
                "name": d,
                "path": dir_rel,
                "is_dir": True,
                "size": 0,
                "extension": "",
            })

        for f in files:
            ext = os.path.splitext(f)[1].lower()
            rel_path = os.path.join(rel_root, f) if rel_root else f
            abs_path = os.path.join(root, f)
            size = os.path.getsize(abs_path)

            items.append({
                "name": f,
                "path": rel_path,
                "is_dir": False,
                "size": size,
                "extension": ext,
                "is_build_artifact": ext in ignored_exts,
            })

    # Sort: folders first, then files alphabetically
    items.sort(key=lambda x: (not x["is_dir"], x["path"].lower()))
    return {"files": items}


@app.get("/api/file")
def get_file(path: str = Query(..., description="Relative path from report root")):
    target = (REPORT_DIR / path).resolve()
    if not str(target).startswith(str(REPORT_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")

    ext = target.suffix.lower()
    image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}

    if ext in image_exts:
        return FileResponse(target)

    # Text files
    try:
        content = target.read_text(encoding="utf-8")
        return {
            "path": path,
            "name": target.name,
            "content": content,
            "is_image": False,
            "size": target.stat().st_size,
        }
    except UnicodeDecodeError:
        return {
            "path": path,
            "name": target.name,
            "content": "[Binary file - cannot display in text editor]",
            "is_image": False,
            "is_binary": True,
            "size": target.stat().st_size,
        }


@app.post("/api/save")
def save_file(req: SaveRequest):
    target = (REPORT_DIR / req.path).resolve()
    if not str(target).startswith(str(REPORT_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(req.content, encoding="utf-8")
    return {"status": "saved", "path": req.path, "size": target.stat().st_size}


@app.post("/api/create-file")
def create_file(req: CreateFileRequest):
    target = (REPORT_DIR / req.path).resolve()
    if not str(target).startswith(str(REPORT_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")
    if target.exists():
        raise HTTPException(status_code=400, detail="File already exists")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(req.content or "", encoding="utf-8")
    return {"status": "created", "path": req.path}


@app.post("/api/delete-file")
def delete_file(req: DeleteFileRequest):
    target = (REPORT_DIR / req.path).resolve()
    if not str(target).startswith(str(REPORT_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"status": "deleted", "path": req.path}


@app.post("/api/upload")
async def upload_file(
    target_dir: str = Form("figures"),
    file: UploadFile = File(...)
):
    dest_dir = (REPORT_DIR / target_dir).resolve()
    if not str(dest_dir).startswith(str(REPORT_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)

    rel_path = os.path.relpath(dest_path, REPORT_DIR)
    return {
        "status": "uploaded",
        "filename": file.filename,
        "path": rel_path,
        "size": len(content),
    }


def parse_tex_log(log_text: str) -> Dict[str, Any]:
    errors = []
    warnings = []
    page_count = None

    # Find page count e.g. Output written on main.pdf (34 pages, 2217265 bytes).
    page_match = re.search(r"Output written on .*?\((\d+)\s+pages?", log_text)
    if page_match:
        page_count = int(page_match.group(1))

    # Parse errors
    lines = log_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("!"):
            err_msg = line[1:].strip()
            err_line_no = None
            err_context = []
            j = i + 1
            while j < len(lines) and j < i + 8:
                next_line = lines[j]
                line_match = re.search(r"^l\.(\d+)", next_line)
                if line_match:
                    err_line_no = int(line_match.group(1))
                    err_context.append(next_line)
                    break
                elif next_line.startswith("!") or next_line.strip() == "":
                    break
                else:
                    err_context.append(next_line)
                j += 1
            errors.append({
                "message": err_msg,
                "line": err_line_no,
                "context": "\n".join(err_context),
            })
        elif "Warning:" in line:
            warnings.append(line.strip())
        i += 1

    return {
        "errors": errors,
        "warnings": warnings,
        "page_count": page_count,
    }


@app.post("/api/compile")
def compile_latex(req: CompileRequest):
    main_file = req.main_file or "main.tex"
    target_main = REPORT_DIR / main_file

    if not target_main.exists():
        raise HTTPException(status_code=404, detail=f"{main_file} not found in {REPORT_DIR}")

    passes = max(1, min(req.passes or 1, 3))
    engine = req.engine or "pdflatex"

    start_time = time.time()
    all_output = []
    success = True

    for p in range(passes):
        cmd = [engine, "-interaction=nonstopmode", main_file]
        try:
            res = subprocess.run(
                cmd,
                cwd=str(REPORT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=45,
            )
            out_str = res.stdout.decode("utf-8", errors="replace") if res.stdout else ""
            all_output.append(f"--- Pass {p+1}/{passes} ---")
            all_output.append(out_str)
            if res.returncode != 0:
                success = False
                break
        except subprocess.TimeoutExpired:
            all_output.append(f"\n[Error] Biên dịch timeout sau 45s ở pass {p+1}")
            success = False
            break
        except Exception as e:
            all_output.append(f"\n[Error] Exception during compilation: {str(e)}")
            success = False
            break

    elapsed = round(time.time() - start_time, 2)
    combined_log = "\n".join(all_output)

    # Also read main.log if it exists for deeper diagnostics
    log_file = REPORT_DIR / f"{Path(main_file).stem}.log"
    diagnostics = {"errors": [], "warnings": [], "page_count": None}
    if log_file.exists():
        try:
            diagnostics = parse_tex_log(log_file.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass

    main_pdf = REPORT_DIR / f"{Path(main_file).stem}.pdf"
    pdf_exists = main_pdf.exists()

    return {
        "success": success and pdf_exists,
        "time_taken": elapsed,
        "passes": passes,
        "page_count": diagnostics.get("page_count"),
        "error_count": len(diagnostics.get("errors", [])),
        "warning_count": len(diagnostics.get("warnings", [])),
        "errors": diagnostics.get("errors", []),
        "warnings": diagnostics.get("warnings", [])[:30],  # cap warnings
        "pdf_url": f"/api/pdf?t={int(time.time() * 1000)}" if pdf_exists else None,
        "raw_output_tail": "\n".join(combined_log.splitlines()[-60:]),
    }


@app.get("/api/pdf")
def get_pdf(t: Optional[str] = None):
    main_pdf = REPORT_DIR / "main.pdf"
    if not main_pdf.exists():
        raise HTTPException(status_code=404, detail="PDF has not been compiled yet")
    return FileResponse(
        main_pdf,
        media_type="application/pdf",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Content-Disposition": "inline; filename=main.pdf",
        },
    )


@app.get("/api/download-pdf")
def download_pdf():
    main_pdf = REPORT_DIR / "main.pdf"
    if not main_pdf.exists():
        raise HTTPException(status_code=404, detail="PDF has not been compiled yet")
    return FileResponse(
        main_pdf,
        media_type="application/pdf",
        filename="report.pdf",
    )


@app.get("/api/raw-log")
def get_raw_log():
    log_file = REPORT_DIR / "main.log"
    if not log_file.exists():
        return {"content": "Chưa có file log. Hãy bấm Biên dịch (Recompile)."}
    try:
        return {"content": log_file.read_text(encoding="utf-8", errors="replace")}
    except Exception as e:
        return {"content": f"Lỗi đọc file log: {str(e)}"}


# Mount static assets
static_path = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/")
def index():
    return FileResponse(static_path / "index.html")


def main():
    parser = argparse.ArgumentParser(description="Local Overleaf LaTeX Studio")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_REPORT_DIR), help="Path to report folder")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8500, help="Port to serve web app")
    args = parser.parse_args()

    target_dir = Path(args.dir).resolve()
    if not target_dir.exists():
        print(f"Directory {target_dir} does not exist. Creating it...")
        target_dir.mkdir(parents=True, exist_ok=True)

    set_report_dir(target_dir)
    print(f"\n=======================================================")
    print(f"  Local LaTeX Studio (Overleaf Style)")
    print(f"  Thư mục làm việc: {REPORT_DIR}")
    print(f"  Địa chỉ truy cập: http://localhost:{args.port}")
    print(f"=======================================================\n")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
