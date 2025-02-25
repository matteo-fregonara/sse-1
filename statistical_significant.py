import re
import numpy as np
import pandas as pd
import scipy.stats as stats

# 1. Specify your file path here
file_path = r"C:\Users\gyumc\OneDrive\바탕 화면\sse1\sse-1\sse_project_1\results_summary.txt"

# 2. Regex to capture Joules from lines like:
#    Energy consumption in joules: 59.032958984375 for 3.6229098 sec of execution.
pattern = re.compile(r"Energy consumption in joules:\s*([\d.]+)")

def extract_energy_data(file_path):
    """
    Reads a results_summary.txt line by line,
    extracts energy consumption (in joules),
    and assigns it to the correct framework based on the preceding lines.
    """
    frameworks = ["pytorch", "tensorflow", "jax"]
    energy_data = {fw: [] for fw in frameworks}
    
    current_fw = None
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_lower = line.lower().strip()
            # Identify which framework block we are in
            if "running pytorch training" in line_lower:
                current_fw = "pytorch"
            elif "running tensorflow training" in line_lower:
                current_fw = "tensorflow"
            elif "running jax training" in line_lower:
                current_fw = "jax"
            
            # Look for "Energy consumption in joules: X"
            match = pattern.search(line)
            if match and current_fw in energy_data:
                # Convert the extracted string to float
                joules_value = float(match.group(1))
                energy_data[current_fw].append(joules_value)

    return energy_data

# Extract the data
energy_data = extract_energy_data(file_path)

# Let's print out how many samples we got for each framework
print("Extracted energy samples (count):")
for fw, values in energy_data.items():
    print(f"{fw}: {len(values)} samples")

# Convert each list to a numpy array (convenient for stats tests)
pytorch_vals = np.array(energy_data["pytorch"])
tensorflow_vals = np.array(energy_data["tensorflow"])
jax_vals = np.array(energy_data["jax"])

# Make sure we have enough data to do statistics:
# (at least 2 data points per framework)
if len(pytorch_vals) < 2 or len(tensorflow_vals) < 2 or len(jax_vals) < 2:
    print("Not enough data in one or more frameworks to perform t-tests.")
    exit()

def compare_two_groups(arr1, arr2, label1, label2):
    """
    Returns a dictionary with:
     - Welch's t-test results
     - Student's t-test results
    """
    # Welch’s t-test (equal_var=False)
    welch_stat, welch_p = stats.ttest_ind(arr1, arr2, equal_var=False)

    # Student’s t-test (equal_var=True)
    student_stat, student_p = stats.ttest_ind(arr1, arr2, equal_var=True)

    return {
        "Frameworks": f"{label1} vs {label2}",
        "Welch's t-statistic": welch_stat,
        "Welch's p-value": welch_p,
        "Student's t-statistic": student_stat,
        "Student's p-value": student_p
    }

# Compare PyTorch vs TensorFlow
pt_tf = compare_two_groups(pytorch_vals, tensorflow_vals, "pytorch", "tensorflow")
# Compare PyTorch vs JAX
pt_jax = compare_two_groups(pytorch_vals, jax_vals, "pytorch", "jax")
# Compare TensorFlow vs JAX
tf_jax = compare_two_groups(tensorflow_vals, jax_vals, "tensorflow", "jax")

# 4. Store results in a DataFrame for clarity
results = pd.DataFrame([pt_tf, pt_jax, tf_jax])
print("\nStatistical Significance Results:\n")
print(results)

# 5. Optionally save the results to CSV
results.to_csv("statistical_significance_results.csv", index=False)
print("\nSaved results to 'statistical_significance_results.csv'")