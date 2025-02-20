@echo off
setlocal enabledelayedexpansion

echo Starting training benchmark - 30 iterations
echo ========================================

for /l %%i in (1, 1, 1) do (
    echo.
    echo Running iteration %%i of 30
    echo ----------------------------------------

    echo Running PyTorch training...
    energibridge.exe python train_pytorch.py -o C:\Users\matte\PycharmProjects\sse_project_1\results.csv --summary %*

    echo Waiting 60 seconds...
    timeout /t 1 /nobreak > nul

    echo Running TensorFlow training...
    energibridge.exe python train_tensorflow.py -o C:\Users\matte\PycharmProjects\sse_project_1\results.csv --summary %*

    echo Waiting 60 seconds...
    timeout /t 1 /nobreak > nul
)


echo.
echo ========================================
echo Benchmark completed - all 30 iterations finished
echo Results saved to results.csv

endlocal