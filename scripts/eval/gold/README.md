# SA-Co/Gold benchmark

SA-Co/Gold is a benchmark for promptable concept segmentation (PCS) in images. The benchmark contains images paired with text labels, also referred as Noun Phrases (NPs), each annotated exhaustively with masks on all object instances that match the label. SA-Co/Gold comprises 7 subsets, each targeting a different annotation domain: MetaCLIP captioner NPs, SA-1B captioner NPs, Attributes, Crowded Scenes, Wiki-Common1K, Wiki-Food/Drink, Wiki-Sports Equipment. The images are originally from the MetaCLIP and SA-1B datasets.

For each subset, the annotations are multi-reviewed by 3 independent human annotators. Each row in the figure shows an image and noun phrase pair from
one of the domains, and masks from the 3 annotators. Dashed borders indicate special group masks that cover more than a single instance, used when separating into instances is deemed too difficult. Annotators sometimes disagree on precise mask borders, the number of instances, and whether the phrase exists. Having 3 independent annotations allow us to measure human agreement on the task, which serves as an upper bound for model performance.


<p align="center">
  <img src="../../../assets/saco_gold_annotation.png?" style="width:80%;" />
</p>
# Preparation

## Download annotations

The GT annotations can be downloaded from [Hugging Face](https://huggingface.co/datasets/facebook/SACo-Gold) or [Roboflow](https://universe.roboflow.com/sa-co-gold)

## Download images

There are two image sources for the evaluation dataset: MetaCLIP and SA-1B.

1) The MetaCLIP images are referred in 6 out of 7 subsets (MetaCLIP captioner NPs, Attributes, Crowded Scenes, Wiki-Common1K, Wiki-Food/Drink, Wiki-Sports Equipment) and can be downloaded from [Roboflow](https://universe.roboflow.com/sa-co-gold/gold-metaclip-merged-a-release-test/).

2) The SA-1B images are referred in 1 out of 7 subsets (SA-1B captioner NPs) and can be downloaded from [Roboflow](https://universe.roboflow.com/sa-co-gold/gold-sa-1b-merged-a-release-test/). Alternatively, they can be downloaded from [here](https://ai.meta.com/datasets/segment-anything-downloads/). Please access the link for `sa_co_gold.tar` from dynamic links available under `Download text file` to download the SA-1B images referred in SA-Co/Gold.

# Usage
## Visualization

- Visualize GT annotations: [saco_gold_silver_vis_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/saco_gold_silver_vis_example.ipynb)
- Visualize GT annotations and sample predictions side-by-side: [sam3_data_and_predictions_visualization.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_data_and_predictions_visualization.ipynb)


## Run evaluation

The official metric for SA-Co/Gold is cgF1. Please refer to the SAM3 paper for details.
Our evaluator inherits from the official COCO evaluator, with some modifications. Recall that in the Gold subset, there are three annotations for each datapoint. We evaluate against each of them and picks the most favorable (oracle setting). It has minimal dependency (pycocotools, numpy and scipy), to help reusability in other projects. In this section we provide several pointers to run evaluation of SAM3 or third-party models.

### Evaluate SAM3

We provide inference configurations to reproduce the evaluation of SAM3.
First, please edit the file [eval_base.yaml](https://github.com/facebookresearch/sam3/blob/main/sam3/train/configs/eval_base.yaml) with the paths where you downloaded the images and annotations above.

There are 7 subsets and as many configurations to be run.
Let's take the first subset as an example. The inference can be run locally using the following command (you can adjust the number of gpus):
```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_metaclip_nps.yaml --use-cluster 0 --num-gpus 1
```
The predictions will be dumped in the folder specified in eval_base.yaml.

We also provide support for SLURM-based cluster inference. Edit the eval_base.yaml file to reflect your slurm configuration (partition, qos, ...), then run

```bash
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_metaclip_nps.yaml --use-cluster 1
```

We provide the commands for all subsets below
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

### Offline evaluation

If you have the predictions in the COCO result format (see [here](https://cocodataset.org/#format-results)), then we provide scripts to easily run the evaluation.

For an example on how to run the evaluator on all subsets and aggregate results, see the following notebook: [saco_gold_silver_eval_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/saco_gold_silver_eval_example.ipynb)
Alternatively, you can run `python scripts/eval/gold/eval_sam3.py`

If you have a prediction file for a given subset, you can run the evaluator specifically for that one using the standalone script. Example:
```bash
python scripts/eval/standalone_cgf1.py --pred_file /path/to/coco_predictions_segm.json --gt_files /path/to/annotations/gold_metaclip_merged_a_release_test.json  /path/to/annotations/gold_metaclip_merged_b_release_test.json  /path/to/annotations/gold_metaclip_merged_c_release_test.json
```


# Results
Here we collect the segmentation results for SAM3 and some baselines. Note that the baselines that do not produce masks are evaluated by converting the boxes to masks using SAM2
<table style="border-color:black;border-style:solid;border-width:1px;border-collapse:collapse;border-spacing:0;text-align:right" class="tg"><thead>
<tr><th style="text-align:center"></th><th style="text-align:center" colspan="3">Average</th><th style="text-align:center" colspan="3">Captioner metaclip</th><th style="text-align:center" colspan="3">Captioner sa1b</th>
<th style="text-align:center" colspan="3">Crowded</th><th style="text-align:center" colspan="3">FG food</th><th style="text-align:center" colspan="3">FG sport</th><th style="text-align:center" colspan="3">Attributes</th>
<th style="text-align:center" colspan="3">Wiki common</th></tr>
</thead>
<tbody>
<tr><td ></td><td >cgF1</td><td >IL_MCC</td><td >positive_micro_F1</td>
<td >cgF1</td><td >IL_MCC</td><td >positive_micro_F1</td><td >cgF1</td>
<td >IL_MCC</td><td >positive_micro_F1</td><td >cgF1</td><td >IL_MCC</td>
<td >positive_micro_F1</td><td >cgF1</td><td >IL_MCC</td><td >positive_micro_F1</td>
<td >cgF1</td><td >IL_MCC</td><td >positive_micro_F1</td><td >cgF1</td>
<td >IL_MCC</td><td >positive_micro_F1</td><td >cgF1</td><td >IL_MCC</td>
<td >positive_micro_F1</td></tr>
<tr><td >gDino-T</td><td >3.25</td><td >0.15</td><td >16.2</td>
<td >2.89</td><td >0.21</td><td >13.88</td><td >3.07</td>
<td >0.2</td><td >15.35</td><td >0.28</td><td >0.08</td>
<td >3.37</td><td >0.96</td><td >0.1</td><td >9.83</td>
<td >1.12</td><td >0.1</td><td >11.2</td><td >13.75</td>
<td >0.29</td><td >47.3</td><td >0.7</td><td >0.06</td>
<td >12.14</td></tr>
<tr><td >OWLv2*</td><td >24.59</td><td >0.57</td><td >42</td>
<td >17.69</td><td >0.52</td><td >34.27</td><td >13.32</td>
<td >0.5</td><td >26.83</td><td >15.8</td><td >0.51</td>
<td >30.74</td><td >31.96</td><td >0.65</td><td >49.35</td>
<td >36.01</td><td >0.64</td><td >56.19</td><td >35.61</td>
<td >0.63</td><td >56.23</td><td >21.73</td><td >0.54</td>
<td >40.25</td></tr>
<tr><td >OWLv2</td><td >17.27</td><td >0.46</td><td >36.8</td>
<td >12.21</td><td >0.39</td><td >31.33</td><td >9.76</td>
<td >0.45</td><td >21.65</td><td >8.87</td><td >0.36</td>
<td >24.77</td><td >24.36</td><td >0.51</td><td >47.85</td>
<td >24.44</td><td >0.52</td><td >46.97</td><td >25.85</td>
<td >0.54</td><td >48.22</td><td >15.4</td><td >0.42</td>
<td >36.64</td></tr>
<tr><td >LLMDet-L</td><td >6.5</td><td >0.21</td><td >27.3</td>
<td >4.49</td><td >0.23</td><td >19.36</td><td >5.32</td>
<td >0.23</td><td >22.81</td><td >2.42</td><td >0.18</td>
<td >13.74</td><td >5.5</td><td >0.19</td><td >29.12</td>
<td >4.39</td><td >0.17</td><td >25.34</td><td >22.17</td>
<td >0.39</td><td >57.13</td><td >1.18</td><td >0.05</td>
<td >23.3</td></tr>
<tr><td >APE</td><td >16.41</td><td >0.4</td><td >36.9</td>
<td >12.6</td><td >0.42</td><td >30.11</td><td >2.23</td>
<td >0.22</td><td >10.01</td><td >7.15</td><td >0.35</td>
<td >20.3</td><td >22.74</td><td >0.51</td><td >45.01</td>
<td >31.79</td><td >0.56</td><td >56.45</td><td >26.74</td>
<td >0.47</td><td >57.27</td><td >11.59</td><td >0.29</td>
<td >39.46</td></tr>
<tr><td >DINO-X</td><td >21.26</td><td >0.38</td><td >55.2</td>
<td >17.21</td><td >0.35</td><td >49.17</td><td >19.66</td>
<td >0.48</td><td >40.93</td><td >12.86</td><td >0.34</td>
<td >37.48</td><td >30.07</td><td >0.49</td><td >61.72</td>
<td >28.36</td><td >0.41</td><td >69.4</td><td >30.97</td>
<td >0.42</td><td >74.04</td><td >9.72</td><td >0.18</td>
<td >53.52</td></tr>
<tr><td >Gemini 2.5</td><td >13.03</td><td >0.29</td><td >46.1</td>
<td >9.9</td><td >0.29</td><td >33.79</td><td >13.1</td>
<td >0.41</td><td >32.1</td><td >8.15</td><td >0.27</td>
<td >30.34</td><td >19.63</td><td >0.33</td><td >59.52</td>
<td >15.07</td><td >0.28</td><td >53.5</td><td >18.84</td>
<td >0.3</td><td >63.14</td><td >6.5</td><td >0.13</td>
<td >50.32</td></tr>
<tr><td >SAM 3</td><td >54.06</td><td >0.82</td><td >66.11</td>
<td >47.26</td><td >0.81</td><td >58.58</td><td >53.69</td>
<td >0.86</td><td >62.55</td><td >61.08</td><td >0.9</td>
<td >67.73</td><td >53.41</td><td >0.79</td><td >67.28</td>
<td >65.52</td><td >0.89</td><td >73.75</td><td >54.93</td>
<td >0.76</td><td >72</td><td >42.53</td><td >0.7</td>
<td >60.85</td></tr>
</tbody></table>



# Annotation format

The annotation format is derived from [COCO format](https://cocodataset.org/#format-data). Notable data fields are:

- `images`: a `list` of `dict` features, contains a list of all image-NP pairs. Each entry is related to an image-NP pair and has the following items.
  - `id`: an `int` feature, unique identifier for the image-NP pair
  - `text_input`: a `string` feature, the noun phrase for the image-NP pair
  - `file_name`: a `string` feature, the relative image path in the corresponding data folder.
  - `height`/`width`: dimension of the image
  - `is_instance_exhaustive`: Boolean (0 or 1). If it's 1 then all the instances are correctly annotated. For instance segmentation, we only use those datapoints. Otherwise, there may be either missing instances or crowd segments (a segment covering multiple instances)
  - `is_pixel_exhaustive`: Boolean (0 or 1). If it's 1, then the union of all masks cover all pixels corresponding to the prompt. This is weaker than instance_exhaustive since it allows crowd segments. It can be used for semantic segmentation evaluations.

- `annotations`: a `list` of `dict` features, containing a list of all annotations including bounding box, segmentation mask, area etc.
  - `image_id`: an `int` feature, maps to the identifier for the image-np pair in images
  - `bbox`: a `list` of float features, containing bounding box in [x,y,w,h] format, normalized by the image dimensions
  - `segmentation`: a dict feature, containing segmentation mask in RLE format
  - `category_id`: For compatibility with the coco format. Will always be 1 and is unused.
  - `is_crowd`: Boolean (0 or 1). If 1, then the segment overlaps several instances (used in cases where instances are not separable, for e.g. due to poor image quality)

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

# Data Stats

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
