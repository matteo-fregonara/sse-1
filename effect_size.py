import numpy as np
from scipy.stats import shapiro

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re


pattern = re.compile(r"Energy consumption in joules:\s*([\d.]+)")


def extract_data(file_path):
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


def mean_difference(s1, s2):
    return np.mean(s1) - np.mean(s2)


def percent_change(s1, s2):
    return (mean_difference(s1, s2) / np.mean(s1)) * 100


def cohens_d(s1, s2):
    return mean_difference(s1, s2) / np.sqrt((np.std(s1) ** 2 + np.std(s2) ** 2) / 2)


def shapiro_wilk_test(s):
    stat, p = shapiro(s)
    return stat, p


def plot_violin(pytorch_data, tensorflow_data, jax_data):
    # Data
    frameworks = ["Pytorch", "TensorFlow", "Jax"]
    energy_data = {
        "Pytorch": pytorch_data,
        "TensorFlow": tensorflow_data,
        "Jax": jax_data
    }

    # Convert data into DataFrame for Seaborn
    df = pd.DataFrame([(fw, value) for fw, values in energy_data.items() for value in values],
                      columns=["Framework", "Energy Consumption (Joules)"])

    # Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Framework", y="Energy Consumption (Joules)", data=df, inner="box", palette="muted")

    # Labels and Title
    plt.title("Energy Consumption Distribution of Different Frameworks", fontsize=14)
    plt.xlabel("Framework", fontsize=12)
    plt.ylabel("Energy Consumption (Joules)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Show plot
    plt.show()


if __name__ == "__main__":
    joules = extract_data("sse_project_1/results_summary.txt")
    pytorch_data = joules["pytorch"]
    tensorflow_data = joules["tensorflow"]
    jax_data = joules["jax"]

    print("[Pytorch]:", pytorch_data)
    print("[TensorFlow]:", tensorflow_data)
    print("[Jax]:", jax_data)
    print()
    print("Mean [Pytorch]:", np.mean(pytorch_data))
    print("Mean [TensorFlow]:", np.mean(tensorflow_data))
    print("Mean [Jax]:", np.mean(jax_data))
    print()
    print("Mean Difference [Pytorch - Tensorflow]:", mean_difference(pytorch_data, tensorflow_data))
    print("Mean Difference [Pytorch - Jax]:", mean_difference(pytorch_data, jax_data))
    print("Mean Difference [Tensorflow - Jax]:", mean_difference(tensorflow_data, jax_data))
    print()
    print("Percent Change [Pytorch - Tensorflow]:", percent_change(pytorch_data, tensorflow_data))
    print("Percent Change [Pytorch - Jax]:", percent_change(pytorch_data, jax_data))
    print("Percent Change [Tensorflow - Jax]:", percent_change(tensorflow_data, jax_data))
    print()
    print("Cohens [Pytorch - Tensorflow]:", cohens_d(pytorch_data, tensorflow_data))
    print("Cohens [Pytorch - Jax]:", cohens_d(pytorch_data, jax_data))
    print("Cohens [Tensorflow - Jax]:", cohens_d(tensorflow_data, jax_data))
    print()
    print("Shapiro-Wilkinson Test [Pytorch]:", shapiro_wilk_test(pytorch_data)[1])
    print("Shapiro-Wilkinson Test [Tensorflow]:", shapiro_wilk_test(tensorflow_data)[1])
    print("Shapiro-Wilkinson Test [Jax]:", shapiro_wilk_test(jax_data)[1])

    plot_violin(pytorch_data, tensorflow_data, jax_data)
