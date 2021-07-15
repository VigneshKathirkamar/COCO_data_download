from pycocotools.coco import COCO
import sys
import pandas as pd


# Initializations (make them as arguments)
cat_of_interest = "person" # supports only one category.
ann_file = "/home/vvsa/vignesh/coco_dataset/annotations_trainval2017/annotations/instances_val2017.json"
#csv_save_path = "/media/VjSravani/Sravani/public-datasets/coco/annotations/keyboard-trn-bboxes.csv"
csv_save_path = "/home/vvsa/vignesh/coco_dataset/person/coco_person.csv"
# COCO instance
coco = COCO(ann_file)

# get category id
cat_id  = coco.getCatIds(catNms=[cat_of_interest])

# get annotation ids for current category
ann_ids = coco.getAnnIds(catIds=cat_id, iscrowd=None)
all_ann = coco.loadAnns(ann_ids)
 
# Loop through each annotation and create a data frame with necessary
#     information to create csv file. This file later aids in creating
#     tensorflow record.

df_rows = []
for i in range(0, len(all_ann)):
    cur_ann    = all_ann[i]
    cbbox      = cur_ann["bbox"]
    cimg_info  = coco.loadImgs(cur_ann["image_id"])

    if(len(cimg_info) > 1):
        print("ERROR: More than one image got loaded")
        sys.exit(1)
        
    filename   = cimg_info[0]["file_name"]
    cur_class  = cat_of_interest
    width    = cimg_info[0]["width"]
    height   = cimg_info[0]["height"]
    xmin     = int(cbbox[0])
    ymin     = int(cbbox[1])
    xmax     = min(int(xmin + cbbox[2]), width-1)
    ymax     = min(int(ymin + cbbox[3]), height-1)

    df_rows  = df_rows + [[filename, str(width), str(height), cur_class,
                           str(xmin), str(ymin), str(xmax), str(ymax)]]

df=pd.DataFrame(df_rows, columns=["filename", "width", "height", "class",
                           "xmin", "ymin", "xmax", "ymax"])
df.to_csv(csv_save_path)

