# SA-CO/Gold benchmark

SA-Co/Gold is a benchmark for promptable concept segmentation (PCS) in images. The benchmark contains images paired with text labels (also referred as Noun Phrases aka NPs), each annotated exhaustively with masks on all object instances that match the label. SA-Co/Gold comprises 7 subsets, each targeting a different annotation domain. For each subset, the annotations are multi-reviewed and agreed by 3 human annotators resulting in a high-quality benchmark.

This dataset covers 2 image sources and 7 annotation domains. The image sources are: MetaCLIP and SA-1B. The annotation domains are: MetaCLIP captioner NPs, SA-1B captioner NPs, Attributes, Crowded Scenes, Wiki-Common1K, Wiki-Food/Drink, Wiki-Sports Equipment.

## Usage

### Download annotations

The GT annotations can be downloaded from [Hugging Face](https://huggingface.co/datasets/facebook/SACo-Gold) or [Roboflow](https://sa-co-gold.roboflow.com/gt-annotations.zip)

### Download images

There are two image sources for the evaluation dataset: MetaCLIP and SA-1B.

1) The MetaCLIP images are referred in 6 out of 7 subsets (MetaCLIP captioner NPs, Attributes, Crowded Scenes, Wiki-Common1K, Wiki-Food/Drink, Wiki-Sports Equipment) and can be downloaded from [Roboflow](https://sa-co-gold.roboflow.com/metaclip-images.zip).

2) The SA-1B images are referred in 1 out of 7 subsets (SA-1B captioner NPs) and can be downloaded from the publicly released [version](https://ai.meta.com/datasets/segment-anything-downloads/). Please access the link for `sa_co_gold.tar` from dynamic links available under `Download text file` to download the SA-1B images referred in SA-Co/Gold.

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

Details on the annotation format can be found on [Hugging Face](https://huggingface.co/datasets/facebook/SACo-Gold#annotation-format).
