# Evaluation dataset

SA-Co/Gold is an evaluation dataset comprising of 7 subsets, each targeting a different scenario. For each subset, the annotations are multi-reviewed and agreed by 3 human annotators resulting in a high quality test set.

## Usage

### Download annotations

The GT annotations can be downloaded from the following [location](https://drive.google.com/drive/folders/1_DzesohTS4WYoKdPVLy2cY-UfJhcOjE2?usp=sharing)

### Download images

There are two image sources for the evaluation dataset: MetaCLIP and SA-1B.

1) The MetaCLIP images are referred in 6 out of 7 subsets and can be downloaded using the below commond. This will download all the images from the urls referred in `gold_metaclip_filename_urls_mapping_release.json`. The script should download 14856 images. (**Note for Roboflow:** The downloaded version with 14810 images i.e. 46 missing images is also fine and the annotations are updated accordingly)

```bash
python download_metaclip_urls.py
```

2) The SA-1B images referred in `sa1b_filenames.txt` (997 images) can be downloaded from the publicly released version. Please access the link for `sa_co_gold.tar` from dynamic links available under `Download text file` option in the publicly released [version](https://ai.meta.com/datasets/segment-anything-downloads/) to download and extract the SA-1B images.


### Run online evaluation

Update the path for GT annotation and images and run the below command for online evaluation of 7 subsets.

#### MetaCLIP captioner NPs

```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_metaclip_nps.yaml --use-cluster 1
```
#### SA-1B captioner NPs

Refer to SA-1B images for this subset. For the other 6 subsets, refer to MetaCLIP images.

```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_sa1b_nps.yaml --use-cluster 1
```
#### Attributes

```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_attributes.yaml --use-cluster 1
```
#### Crowded Scenes

```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_crowded.yaml --use-cluster 1
```
#### Wiki-Common1K

```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_wiki_common.yaml --use-cluster 1
```
#### Wiki-Food/Drink

```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_fg_food.yaml --use-cluster 1
```

#### Wiki-Sports Equipment

```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_fg_sports.yaml --use-cluster 1
```

## Annotation format

The annotation format is derived from [COCO format](https://cocodataset.org/#format-data). 

Here are few additional details about GT annotations in each subset of SA-Co/Gold.
1) "images" contain list of all image-noun phrase pairs. Each entry has a unique "id", the noun phrase is within key "text_input" and the image path (relative to root folder) is within key "file_name".
2) "annotations" contain list of all annotations including bbox, segmentation mask, area etc. The "image_id" here maps to the "id" in "images". <br> For "id" in "images" that don't have any annotations (i.e. they do not exist in any of the "image_id" in "annotations"), we refer to them as <b> negative </b> image-noun phrase pair. And, for "id" in "images" that have corresponding annotations (i.e. exist as  "image_id" in "annotations"), we refer to them as <b> positive </b> image-noun phrase pair.
3) "category" contains only 1 category since we evaluate instance segmentation.

For predictions, the json file should contain list of all predicted annotations. The keys are similar to the GT annotations and in addition, it should have "score" that stores the confidence of model prediction.
