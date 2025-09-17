# Evaluation dataset

SAC-Gold is an evaluation dataset comprising of 7 subsets, each targeting a different scenario. For each subset, the annotations are multi-reviewed and agreed by 3 human annotators resulting in a high quality test set.

## Usage

### Download annotations

The GT annotations can be downloaded from the following [location](https://drive.google.com/drive/folders/1o4IKrf3BvFMmUabpgZdZVDB5b7xG0CWB?usp=drive_link)

### Download images

There are two image sources for the evaluation dataset: MetaCLIP and SA-1B.

1) The MetaCLIP images are referred in 6 out of 7 subsets and can be downloaded using the below commond. This will download all the images from the urls referred in `gold_metaclip_filename_urls_mapping.json`. The script should download 22507 images.

```bash
python download_metaclip_urls_submitit.py
```

2) The SA-1B images can be referred from the publicly released version.

### Update annotations (Optional)

If the number of MetaCLIP images downloaded is less than 22507, then run the below script to update the groundtruth annotations. This will filter annotations related to missing images.
```bash
python update_eval_files.py
```

### Run evaluation

Sample SAM3 model predictions on SAC-Gold test set can download them from the following [location](https://drive.google.com/drive/folders/12-H7PwNCRkXU_7vJK_2kTSRR15ZiZ0Yt?usp=drive_link)

Run the below command for offline evaluation.

```bash
python eval_sam3.py
```

The above script will print the below after running the evaluation.
```
Subset name, CG_F1, IL_MCC, pmF1, demoF1, J&F
epsilon: 55.52,0.77,71.87,52.96,84.61
sa1b: 55.69,0.81,69.07,56.17,81.75
crowded: 53.31,0.83,64.45,61.66,83.74
fg_food: 65.29,0.83,78.72,66.15,87.8
fg_sports_equipment: 69.76,0.89,78.03,65.23,90.29
attributes: 56.79,0.67,84.3,58.02,89.23
wiki_common: 46.5,0.6,77.09,51.69,86.69
```

## Annotation format

The annotation format is derived from [COCO format](https://cocodataset.org/#format-data). 

Here are few additional details about GT annotations in each subset of SAC-Gold.
1) "images" contain list of all image-noun phrase pairs. Each entry has a unique "id", the noun phrase is within key "text_input" and the image path is within key "file_name".
2) "annotations" contain list of all annotations including bbox, segmentation mask, area etc. The "image_id" here maps to the "id" in "images".
3) "category" contains only 1 category since we evaluate instance segmentation.

For predictions, the json file should contain list of all predicted annotations. The keys are similar to the GT annotations, in addition it should have "score" that stores the confidence of model prediction.
