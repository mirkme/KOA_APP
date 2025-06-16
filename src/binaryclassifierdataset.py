import os

# Set the path to your dataset directory
folder_path = "C:\\Users\\Asus\\Desktop\\selfdata\\other"

# Get list of all files in the directory (sorted for consistency)
files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

# Loop through files and rename them
for index, filename in enumerate(files, start=1):
    # Get the file extension (e.g., .jpg, .png)
    ext = os.path.splitext(filename)[1]
    
    # Set new filename (e.g., image1.jpg)
    new_name = f"image{index}{ext}"
    
    # Full old and new paths
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    
    # Rename the file
    os.rename(old_path, new_path)

print("Renaming complete!")