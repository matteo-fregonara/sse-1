## Setup Instructions

Create and start the RAPL service before running any benchmarks:
   - Open a Command Prompt window in Administrator mode
   - First, create the RAPL service by running:
     ```
     sc create rapl type=kernel binPath="<absolute_path_to_LibreHardwareMonitor.sys>"
     ```
     Replace `<absolute_path_to_LibreHardwareMonitor.sys>` with the actual path to the driver file
   - Start the service:
     ```
     sc start rapl
     ```
   - Verify the service is running with:
     ```
     sc query rapl
     ```
     
2. Install all required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Experiment

1. Open a Command Prompt window with **Admin privileges**

2. Navigate to the repository directory:

3. Run the benchmark script:
   ```
   .\run_benchmark.bat
   ```

4. The energy consumption in Joules will be printed in the terminal and the detailed results will be saved in the `results` directory.

