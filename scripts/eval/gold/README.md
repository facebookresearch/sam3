# Evaluation dataset

SA-Co/Gold is an evaluation dataset comprising of 7 subsets, each targeting a different scenario. For each subset, the annotations are multi-reviewed and agreed by 3 human annotators resulting in a high quality test set.

## Usage

### Download annotations

The GT annotations can be downloaded from the following [location](https://drive.google.com/drive/folders/1jT8Bsvrnz8nOqtksQjLlwrUcYFjO1kH8?usp=sharing)

### Download images

There are two image sources for the evaluation dataset: MetaCLIP and SA-1B.

1) The MetaCLIP images are referred in 6 out of 7 subsets and can be downloaded using the below commond. This will download all the images from the urls referred in `gold_metaclip_filename_urls_mapping_release.json`. The script should download 14856 images.

```bash
python download_metaclip_urls.py
```

2) The SA-1B images can be referred from the publicly released version.

### Update annotations (Optional)

If the number of MetaCLIP images downloaded is less than 14856, then run the below script to update the groundtruth annotations. This will filter annotations related to missing images.
```bash
python update_eval_files.py
```

### Evaluation

#### Run offline evaluation

Sample SAM3 model predictions on SA-Co/Gold test set can be download from the following [location](https://drive.google.com/drive/folders/1-3fJCIYhB4ugN7SIPkn-OrEeXYvowLUN?usp=sharing)

Run the below command for offline evaluation.

```bash
python eval_sam3.py
```

The above script will print metrics similar to below after running the evaluation.
```
Subset name, CG_F1, IL_MCC, pmF1, demoF1, J&F
metaclip: 61.25,0.81,75.93,55.82,84.82
sa1b: 64.55,0.86,75.06,61.94,83.06
crowded: 60.72,0.88,69.08,64.39,83.91
fg_food: 66.59,0.81,82.66,62.24,87.22
fg_sports_equipment: 74.73,0.9,82.98,70.61,90.27
attributes: 66.44,0.77,86.69,64.27,88.51
wiki_common: 56.03,0.69,80.77,57.22,85.65
```

#### Run online evaluation

Update the path for GT annotation and images and run the below command for online evaluation.

```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_attributes.yaml --use-cluster 1
```

## Annotation format

The annotation format is derived from [COCO format](https://cocodataset.org/#format-data). 

Here are few additional details about GT annotations in each subset of SA-Co/Gold.
1) "images" contain list of all image-noun phrase pairs. Each entry has a unique "id", the noun phrase is within key "text_input" and the image path (relative to root folder) is within key "file_name".
2) "annotations" contain list of all annotations including bbox, segmentation mask, area etc. The "image_id" here maps to the "id" in "images". <br> For "id" in "images" that don't have any annotations (i.e. they do not exist in any of the "image_id" in "annotations"), we refer to them as <b> negative </b> image-noun phrase pair. And, for "id" in "images" that have corresponding annotations (i.e. exist as  "image_id" in "annotations"), we refer to them as <b> positive </b> image-noun phrase pair.
3) "category" contains only 1 category since we evaluate instance segmentation.

For predictions, the json file should contain list of all predicted annotations. The keys are similar to the GT annotations and in addition, it should have "score" that stores the confidence of model prediction.
