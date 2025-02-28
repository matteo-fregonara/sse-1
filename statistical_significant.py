import re
import numpy as np
import pandas as pd
import scipy.stats as stats

file_path = r"C:\Users\gyumc\OneDrive\바탕 화면\sse1\sse-1\sse_project_1\results_summary.txt"

pattern = re.compile(r"Energy consumption in joules:\s*([\d.]+)")

def extract_energy_data(file_path):
    frameworks = ["pytorch", "tensorflow", "jax"]
    energy_data = {fw: [] for fw in frameworks}
    
    current_fw = None
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_lower = line.lower().strip()
            if "running pytorch training" in line_lower:
                current_fw = "pytorch"
            elif "running tensorflow training" in line_lower:
                current_fw = "tensorflow"
            elif "running jax training" in line_lower:
                current_fw = "jax"
            
            match = pattern.search(line)
            if match and current_fw in energy_data:
                joules_value = float(match.group(1))
                energy_data[current_fw].append(joules_value)

    return energy_data

energy_data = extract_energy_data(file_path)

print("Extracted energy samples (count):")
for fw, values in energy_data.items():
    print(f"{fw}: {len(values)} samples")

pytorch_vals = np.array(energy_data["pytorch"])
tensorflow_vals = np.array(energy_data["tensorflow"])
jax_vals = np.array(energy_data["jax"])

if len(pytorch_vals) < 2 or len(tensorflow_vals) < 2 or len(jax_vals) < 2:
    print("Not enough data in one or more frameworks to perform a statistical test.")
    exit()

def compare_whitney(arr1, arr2, label1, label2):
    u_statistic, p_value = stats.mannwhitneyu(arr1, arr2, alternative='two-sided')
    return {
        "Frameworks": f"{label1} vs {label2}",
        "Mann-Whitney U": u_statistic,
        "p-value": p_value
    }

pt_tf = compare_whitney(pytorch_vals, tensorflow_vals, "pytorch", "tensorflow")
pt_jax = compare_whitney(pytorch_vals, jax_vals, "pytorch", "jax")
tf_jax = compare_whitney(tensorflow_vals, jax_vals, "tensorflow", "jax")

results = pd.DataFrame([pt_tf, pt_jax, tf_jax])
print("\nNon-parametric Mann-Whitney Results:\n")
print(results)

results.to_csv("mann_whitney_results.csv", index=False)
print("\nSaved Mann-Whitney results to 'mann_whitney_results.csv'")
