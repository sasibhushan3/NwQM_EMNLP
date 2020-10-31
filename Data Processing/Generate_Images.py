''' This code generates the screenshots of wikipedia pages using the URLs
'''
import imgkit as ik
import os
import pandas as pd
import argparse
from tensorflow.keras.preprocessing import image


# Command line arguments
parser = argparse.ArgumentParser(description='Read Arguments')
parser.add_argument('--dataset_path', type=str, nargs='?', default='wikipages.csv',
                                        help='dataset path')
parser.add_argument('--url_path', type=str, nargs='?', default='url.csv',
                                        help='URLs file path')
parser.add_argument('--images_path', type=str, nargs='?', default='wiki_images/',
                                        help='path of the folder to save the images of the pages')

# Read the Dataset
df = pd.read_csv(args.dataset_path)
df2 = pd.read_csv(args.url_path)


# Keep the URLs of all pages in a list
list2 = []
for i in range(len(df2)):
    list2.append(df2['URL'][i])


c= 0
l1 = []
l2 = []
for i in list2:
    name = df['Name'][c]
    c=c+1
    try:
        # this creates high pixel image
        ik.from_url(i,name+'.jpg')
    except:
        continue
    img_path = name+'.jpg'
    # So we reduce the pixels by using keras libraries
    img = image.load_img(img_path)
    img.save('reduced_images/'+name+'.jpg')
    os.system('rm '+name+'.jpg')
