@echo off
REM One-time setup of the npxsort environment. Run from an Anaconda/Miniforge Prompt.
cd /d "%~dp0"

call conda env create -f environment.yml || goto :fail
call conda activate npxsort || goto :fail

REM CUDA build of torch first (plain pip torch on Windows is CPU-only).
REM If nvidia-smi reports CUDA older than 12.6, replace cu126 with cu118.
pip install torch --index-url https://download.pytorch.org/whl/cu126 || goto :fail
pip install -r requirements.txt || goto :fail

python check_env.py || goto :fail
echo.
echo Setup finished. Next: run_e097.bat
exit /b 0

:fail
echo.
echo Setup stopped at the step above. Copy the error into the chat.
exit /b 1
