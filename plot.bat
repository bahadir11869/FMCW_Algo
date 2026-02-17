@echo off
setlocal

REM Bu .bat'in bulunduğu klasöre geç
cd PLOT_MAP

if %errorlevel%==0 (
    python plot_all.py
) else (
    py -3 plot_all.py
)

endlocal
