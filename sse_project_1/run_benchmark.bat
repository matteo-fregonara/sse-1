@echo off
setlocal enabledelayedexpansion

:: Create results directory if it doesn't exist
if not exist "results" mkdir results

echo Warming up by running Fibonacci for 5min
python warmup.py
cls
echo Starting training benchmark - 30 iterations
echo ========================================

for /l %%i in (1, 1, 30) do (
    echo.
    echo Running iteration %%i of 30
    echo ----------------------------------------

    :: Get current date and time for unique filenames
    for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
    set "date_time=!dt:~0,8!_!dt:~8,6!"

    :: Create unique filenames for each run
    set "pytorch_result=results\pytorch_iter%%i_!date_time!.csv"
    set "tensorflow_result=results\tensorflow_iter%%i_!date_time!.csv"
    set "jax_result=results\jax_iter%%i_!date_time!.csv"

    echo Running PyTorch training...
    energibridge.exe -o !pytorch_result! --summary python train_pytorch.py
    echo Results saved to !pytorch_result!

    echo Waiting 60 seconds...
    timeout /t 60 /nobreak > nul

    echo Running TensorFlow training...
    energibridge.exe -o !tensorflow_result! --summary python train_tensorflow.py
    echo Results saved to !tensorflow_result!

    echo Waiting 60 seconds...
    timeout /t 60 /nobreak > nul

    echo Running Jax training...
    energibridge.exe -o !jax_result! --summary python train_jax.py
    echo Results saved to !jax_result!

    echo Waiting 60 seconds...
    timeout /t 60 /nobreak > nul
)

echo.
echo ========================================
echo Benchmark completed - all 30 iterations finished
echo Results saved to the 'results' directory

endlocal