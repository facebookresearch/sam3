# SA-Co/Gold benchmark

SA-Co/Gold is a benchmark for promptable concept segmentation (PCS) in images. The benchmark contains images paired with text labels, also referred as Noun Phrases (NPs), each annotated exhaustively with masks on all object instances that match the label. SA-Co/Gold comprises 7 subsets, each targeting a different annotation domain: MetaCLIP captioner NPs, SA-1B captioner NPs, Attributes, Crowded Scenes, Wiki-Common1K, Wiki-Food/Drink, Wiki-Sports Equipment. The images are originally from the MetaCLIP and SA-1B datasets.

For each subset, the annotations are multi-reviewed by 3 human annotators. Each row in the figure shows an image and noun phrase pair from
three of the domains with masks from the 3 annotators overlayed. Dashed borders indicate special group masks that cover more than a single instance, used when separating into instances is deemed too difficult. Annotators sometimes disagree on precise mask borders, the number of instances, and whether the phrase exists. Having 3 independent annotations allow us to measure the human agreement on the task, to serve as an upper bound for model performance.

![SA-Co dataset](assets/saco_gold_annotation.png?raw=true)


## Usage

### Download annotations

The GT annotations can be downloaded from [Hugging Face](https://huggingface.co/datasets/facebook/SACo-Gold) or [Roboflow](https://sa-co.roboflow.com/gold/gt-annotations.zip)

### Download images

There are two image sources for the evaluation dataset: MetaCLIP and SA-1B.

1) The MetaCLIP images are referred in 6 out of 7 subsets (MetaCLIP captioner NPs, Attributes, Crowded Scenes, Wiki-Common1K, Wiki-Food/Drink, Wiki-Sports Equipment) and can be downloaded from [Roboflow](https://sa-co.roboflow.com/gold/metaclip-images.zip).

2) The SA-1B images are referred in 1 out of 7 subsets (SA-1B captioner NPs) and can be downloaded from [Roboflow](https://sa-co.roboflow.com/gold/sa1b-images.zip). Alternatively, they can be downloaded from [here](https://ai.meta.com/datasets/segment-anything-downloads/). Please access the link for `sa_co_gold.tar` from dynamic links available under `Download text file` to download the SA-1B images referred in SA-Co/Gold.

### Visualization

- Visualize GT annotations: [saco_gold_silver_vis_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/saco_gold_silver_vis_example.ipynb)
- Visualize GT annotations and sample predictions side-by-side: [sam3_data_and_predictions_visualization.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_data_and_predictions_visualization.ipynb)


### Run offline evaluation

Update the path for GT annotation and run the below command for offline evaluation of 7 subsets.

```bash
python eval_sam3.py
```

You can also run the following notebook: [saco_gold_silver_eval_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/saco_gold_silver_eval_example.ipynb)

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

The annotation format is derived from [COCO format](https://cocodataset.org/#format-data). Notable data fields are:

- `images`: a `list` of `dict` features, contains a list of all image-NP pairs. Each entry is related to an image-NP pair and has the following items.
  - `id`: a `string` feature, unique identifier for the image-NP pair
  - `text_input`: a `string` feature, the noun phrase for the image-NP pair
  - `file_name`: a `string` feature, the relative image path in the corresponding data folder.

- `annotations`: a `list` of `dict` features, containing a list of all annotations including bounding box, segmentation mask, area etc.
  - `image_id`: a `string` feature, maps to the identifier for the image-np pair in images
  - `bbox`: a `list` of float features, containing bounding box in [x,y,w,h] format
  - `segmentation`: a dict feature, containing segmentation mask in RLE format

- `categories`: a `list` of `dict` features, containing a list of all categories. Here, we provide  the category key for compatibility with the COCO format, but in open-vocabulary detection we do not use it. Instead, the text prompt is stored directly in each image (text_input in images). Note that in our setting, a unique image (id in images) actually corresponds to an (image, text prompt) combination.


For `id` in images that have corresponding annotations (i.e. exist as `image_id` in `annotations`), we refer to them as a "positive" NP. And, for `id` in `images` that don't have any annotations (i.e. they do not exist as `image_id` in `annotations`), we refer to them as a "negative" NP.

A sample annotation from Wiki-Food/Drink domain looks as follows:

#### images

```
[
  {
    "id": 10000000,
    "file_name": "1/1001/metaclip_1_1001_c122868928880ae52b33fae1.jpeg",
    "text_input": "chili",
    "width": 600,
    "height": 600,
    "queried_category": "0",
    "is_instance_exhaustive": 1,
    "is_pixel_exhaustive": 1
  },
  {
    "id": 10000001,
    "file_name": "1/1001/metaclip_1_1001_c122868928880ae52b33fae1.jpeg",
    "text_input": "the fish ball",
    "width": 600,
    "height": 600,
    "queried_category": "2001",
    "is_instance_exhaustive": 1,
    "is_pixel_exhaustive": 1
  }
]
```

#### annotations

```
[
  {
    "id": 1,
    "image_id": 10000000,
    "source": "manual",
    "area": 0.002477777777777778,
    "bbox": [
      0.44333332777023315,
      0.0,
      0.10833333432674408,
      0.05833333358168602
    ],
    "segmentation": {
      "counts": "`kk42fb01O1O1O1O001O1O1O001O1O00001O1O001O001O0000000000O1001000O010O02O001N10001N0100000O10O1000O10O010O100O1O1O1O1O0000001O0O2O1N2N2Nobm4",
      "size": [
        600,
        600
      ]
    },
    "category_id": 1,
    "iscrowd": 0
  },
  {
    "id": 2,
    "image_id": 10000000,
    "source": "manual",
    "area": 0.001275,
    "bbox": [
      0.5116666555404663,
      0.5716666579246521,
      0.061666667461395264,
      0.036666665226221085
    ],
    "segmentation": {
      "counts": "aWd51db05M1O2N100O1O1O1O1O1O010O100O10O10O010O010O01O100O100O1O00100O1O100O1O2MZee4",
      "size": [
        600,
        600
      ]
    },
    "category_id": 1,
    "iscrowd": 0
  }
]
```

### Data Stats

Here are the stats for the 7 annotation domains. The # Image-NPs represent the total number of unique image-NP pairs including both “positive” and “negative” NPs. 


| Domain                   | Media        | # Image-NPs   | # Image-NP-Masks|
|--------------------------|--------------|---------------| ----------------|
| MetaCLIP captioner NPs   | MetaCLIP     | 33393         | 20144           |
| SA-1B captioner NPs      | SA-1B        | 13258         | 30306           |
| Attributes               | MetaCLIP     | 9245          | 3663            |
| Crowded Scenes           | MetaCLIP     | 20687         | 50417           |
| Wiki-Common1K            | MetaCLIP     | 65502         | 6448            |
| Wiki-Food&Drink          | MetaCLIP     | 13951         | 9825            |
| Wiki-Sports Equipment    | MetaCLIP     | 12166         | 5075            |
