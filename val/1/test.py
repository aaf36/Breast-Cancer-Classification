import pandas as pd
import os
import shutil

# Read the CSV file into a DataFrame, skipping the first two rows
df = pd.read_csv("test_list_updated.csv")
print(df)
# Create the "Test Results" subfolder if it doesn't exist
output_folder = "Test_Results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Get the image path and category id
    image_path = row['image'][3:]
    category_id = row['category_id']

    # Check if the image path exists
    if os.path.exists(image_path):
        # Extract the image filename
        image_filename = os.path.basename(image_path)
        # Create the output path for the image in the "Test Results" subfolder
        output_path = os.path.join(output_folder, image_filename)
        
        # Move the image to the "Test Results" subfolder
        shutil.move(image_path, output_path)
        
        print(f"Moved {image_filename} to Test Results folder.")
    else:
        print(f"Image {image_path} does not exist.")
