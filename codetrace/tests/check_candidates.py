import sys
import re

"""
Check progression of mutation candidates by passing in the .out file
"""

def main(file_path):
    try:
        num_candidates=-1
        # Open the file and check for matching lines
        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(r"Collected (\d+) candidates", line)
                if match:
                    num_candidates = int(match.group(1))
                    if num_candidates >= 3500:
                        sys.exit(0)  # Success
        print(f"INFO: not enough num candidates {num_candidates}")
        sys.exit(1)  # No valid line with enough candidates found
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_candidates.py <file_path>")
        sys.exit(1)
    main(sys.argv[1])
