@echo off
REM Sort the e097 intact window. Run from an Anaconda/Miniforge Prompt after setup_env.bat.
REM   run_e097.bat            full run
REM   run_e097.bat motion     stop after preprocessing so you can check motion.png first
cd /d "%~dp0"
call conda activate npxsort || goto :fail

python check_env.py || goto :fail

if /i "%1"=="motion" (
    python run_sorting.py configs\e097_intact_window.yaml --stop-after preprocess || goto :fail
) else (
    python run_sorting.py configs\e097_intact_window.yaml || goto :fail
)
echo.
echo Finished. Results: C:\Users\lpascu\resp_analysis\outputs\e097\intact_window_v2
exit /b 0

:fail
echo.
echo Run stopped with an error. The full log is in outputs\e097\intact_window_v2\logs\
exit /b 1
