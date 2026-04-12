import json
import argparse
from pathlib import Path

import numpy as np


# Function to calculate mean of arrays in nested dictionary
def calculate_means(nested_dict):
    result = {}
    for key1, level1 in nested_dict.items():
        result[key1] = {}
        for key2, level2 in level1.items():
            result[key1][key2] = {}
            for key3, array in level2.items():
                if isinstance(array, list):
                    result[key1][key2][key3] = np.mean(array)
                else:
                    result[key1][key2][key3] = array
    return result


def main():
    parser = argparse.ArgumentParser(description="Average list-valued metrics in a sim json file.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input sim json path.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output averaged json path.")
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)

    means_data = calculate_means(data)

    with open(args.output, 'w') as f:
        json.dump(means_data, f, indent=4)

    print(f"Means calculated and saved to {args.output}")


if __name__ == "__main__":
    main()
