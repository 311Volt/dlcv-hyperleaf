import os
from PIL import Image
import torch
from torchvision import transforms

def transform_tiff(input_file, output_file):
    with Image.open(input_file) as img:
        num_pages = img.n_frames
        transformed_pages = []
        for page_num in range(num_pages):
            img.seek(page_num)
            page = img.copy()
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
            ])
            transformed_pages.append(transform(page))
        transformed_pages[0].save(output_file, save_all=True, append_images=transformed_pages[1:])

input_folder = 'images/'
output_folder = 'images_augmented/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in sorted(os.listdir(input_folder)):
    transform_tiff(f'{input_folder}/{filename}', f'{output_folder}/aug_{filename}')
    print(filename)    
