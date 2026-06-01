@echo off
echo SCRFD 2.5G Face Detection — Quick Start
echo ========================================
echo.
echo 1. Installing dependencies...
pip install -r requirements.txt
echo.
echo 2. Running on test images...
python detect.py test/
echo.
echo Results saved to output/
pause
