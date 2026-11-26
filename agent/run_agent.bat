@echo off
title CVAT 3D Fusion Pre-label Agent (127.0.0.1:8766)
color 0B
cd /d "%~dp0"

echo ====================================================================
echo   CVAT 3D FUSION PRE-LABEL AGENT v1.1 (http://127.0.0.1:8766)
echo   CenterPoint-Pillar + PointPillars-FPN + L-Shape + Camera YOLO11s
echo   100%% Noi bo - Khong dung vao extension cu (cong 8765)
echo ====================================================================
echo.
echo [*] Giai phong cong 8766 neu co tien trinh cu...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8766 ^| findstr LISTENING') do (
    taskkill /f /pid %%a >nul 2>&1
)

if exist "C:\ProgramData\anaconda3\python.exe" (
    set "PY=C:\ProgramData\anaconda3\python.exe"
) else (
    set "PY=python"
)

echo [*] Khoi dong Fusion Agent...
"%PY%" server.py
pause
