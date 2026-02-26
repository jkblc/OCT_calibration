import pickle
import numpy as np
import argparse

def inspect_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"--- Contents of {file_path} ---")

        if isinstance(data, dict):
            for key, value in data.items():
                print(f"Key: '{key}' | Type: {type(value).__name__}")

                if isinstance(value, (list, np.ndarray)):
                    # Convert to numpy array for consistent shape reporting
                    arr = np.array(value)
                    print(f"  Shape: {arr.shape}")
                    # Print first few elements to avoid flooding the terminal
                    print(f"  Preview: {arr.flatten()[:5]} ...")
                else:
                    print(f"  Value: {value}")
                print("-" * 40)
        else:
            # Fallback if the pickle is not a dictionary
            print(data)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading pickle file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a pickle file.")
    parser.add_argument("file", nargs="?", default="calib.pkl", help="Path to the pickle file")
    args = parser.parse_args()

    inspect_pickle(args.file)