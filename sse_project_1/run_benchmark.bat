@echo off
setlocal enabledelayedexpansion

:: Create results.csv header if it doesn't exist
if not exist results.csv (
    echo Creating results.csv file with headers
    echo Iteration,Framework,Timestamp,Command,ExitCode > results.csv
)

echo Starting training benchmark - 30 iterations
echo ========================================

for /l %%i in (1, 1, 1) do (
    echo.
    echo Running iteration %%i of 30
    echo ----------------------------------------

    :: Get current timestamp
    for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
    set "timestamp=!dt:~0,4!-!dt:~4,2!-!dt:~6,2! !dt:~8,2!:!dt:~10,2!:!dt:~12,2!"

    :: Run PyTorch training
    echo Running PyTorch training...
    set "cmd=energibridge.exe python train_pytorch.py %*"
    !cmd!
    set pytorch_exit=%errorlevel%

    :: Log PyTorch result to CSV
    echo %%i,PyTorch,!timestamp!,"!cmd!",!pytorch_exit! >> results.csv
    echo Result logged to results.csv

    echo Waiting 60 seconds...
    timeout /t 1 /nobreak > nul

    :: Get updated timestamp
    for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
    set "timestamp=!dt:~0,4!-!dt:~4,2!-!dt:~6,2! !dt:~8,2!:!dt:~10,2!:!dt:~12,2!"

    :: Run TensorFlow training
    echo Running TensorFlow training...
    set "cmd=energibridge.exe python train_tensorflow.py %*"
    !cmd!
    set tensorflow_exit=%errorlevel%

    :: Log TensorFlow result to CSV
    echo %%i,TensorFlow,!timestamp!,"!cmd!",!tensorflow_exit! >> results.csv
    echo Result logged to results.csv

    echo Waiting 60 seconds...
    timeout /t 1 /nobreak > nul
)

echo.
echo ========================================
echo Benchmark completed - all 30 iterations finished
echo Results saved to results.csv

echo.
echo Contents of results.csv:
type results.csv

endlocal