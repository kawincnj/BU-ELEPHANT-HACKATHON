import os
 
folder_path = r'D:\Code\BU-ELEPHENT-HACKATHON\elephant_img\rus'   # Change this to your folder path
id_value = 1
 
# List all files in the folder
files = os.listdir(folder_path)
 
# Optional: filter only image files (e.g., jpg, png)
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
 
images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
 
# Sort files to keep order consistent
images.sort()
 
for i, filename in enumerate(images, start=1):
    ext = os.path.splitext(filename)[1]
    new_name = f"Rus_{i}{ext}"
    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_name)
    os.rename(old_file, new_file)
 
print("Renaming completed!")