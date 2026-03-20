import os

def clean_yolo_labels(label_dir):
    if not os.path.exists(label_dir):
        print(f"Error: Directory {label_dir} not found.")
        return

    files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    total_removed = 0
    files_modified = 0

    for filename in files:
        file_path = os.path.join(label_dir, filename)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Use a set to keep only unique lines (stripping whitespace)
        unique_lines = list(set(line.strip() for line in lines if line.strip()))

        # Check if duplicates were found
        if len(unique_lines) < len(lines):
            diff = len(lines) - len(unique_lines)
            total_removed += diff
            files_modified += 1
            
            # Overwrite the file with cleaned labels
            with open(file_path, 'w') as f:
                f.write('\n'.join(unique_lines) + '\n')
            print(f"Cleaned {diff} duplicates from: {filename}")

    print(f"\n--- Cleanup Complete ---")
    print(f"Files modified: {files_modified}")
    print(f"Total duplicate lines removed: {total_removed}")

# Update this path to your actual labels directory
label_directory = "/home/lithos_analithics_challenge/images/full_dataset_processed/train/labels/"
clean_yolo_labels(label_directory)