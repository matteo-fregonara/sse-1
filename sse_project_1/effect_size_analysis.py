from extractor import extract_joules
import numpy as np
from scipy.stats import shapiro


def mean_difference(s1, s2):
    return np.mean(s1) - np.mean(s2)


def percent_change(s1, s2):
    return (mean_difference(s1, s2) / np.mean(s1)) * 100


def cohens_d(s1, s2):
    return mean_difference(s1, s2) / np.sqrt((np.std(s1) ** 2 + np.std(s2) ** 2) / 2)


def shapiro_wilk_test(s):
    stat, p = shapiro(s)
    return stat, p


if __name__ == "__main__":
    joules = extract_joules("sse_project_1/results_summary.txt")
    print("Mean Difference:", mean_difference(joules, joules))
    print("Percent Change:", percent_change(joules, joules))
    print("Cohen's d:", cohens_d(joules, joules))
    print("Shapiro-Wilk Test:", shapiro_wilk_test(joules))
