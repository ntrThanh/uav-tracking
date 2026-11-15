#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Đang biên dịch LaTeX bằng pdflatex..."
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "Biên dịch thành công! File output: $(pwd)/main.pdf"
