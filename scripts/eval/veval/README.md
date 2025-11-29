# SA-Co/VEval Dataset
**License** each domain has its own License
* SA-Co/VEval - SA-V: CC-BY-NC 4.0
* SA-Co/VEval - YT-Temporal-1B: CC-BY-NC 4.0
* SA-Co/VEval - SmartGlasses: CC-by-4.0

**SA-Co/VEval** is an evaluation dataset comprising of 3 domains, each domain has a val and test split.
* SA-Co/VEval - SA-V: videos are from the [SA-V dataset](https://ai.meta.com/datasets/segment-anything-video/)
* SA-Co/VEval - YT-Temporal-1B: videos are from the [YT-Temporal-1B](https://cove.thecvf.com/datasets/704)
* SA-Co/VEval - SmartGlasses: egocentric videos from [Smart Glasses](https://huggingface.co/datasets/facebook/SACo-VEval/blob/main/media/saco_sg.tar.gz)

## Environment
Install the SA-Co/VEVal required environment
```
pip install -e ".[veval]"
```
This will allow us to run:
* `scripts/eval/veval/saco_yt1b_downloader.py` preparing frames for SA-Co/VEval - YT-Temporal-1B
* `examples/saco_veval_eval_example.ipynb` example of running an offline evaluator
* `examples/saco_veval_vis_example.ipynb` example of loading and visualizing the data

## Download
### The expected folder structure
The following folder structure is expected after finishing all the download and pre-processing steps in this section
```
data/
├── annotation/
│   ├── saco_veval_sav_test.json
│   ├── saco_veval_sav_val.json
│   ├── saco_veval_smartglasses_test.json
│   ├── saco_veval_smartglasses_val.json
│   ├── saco_veval_yt1b_test.json
│   ├── saco_veval_yt1b_val.json
└── media/
    ├── saco_sav
    │   └── JPEGImages_24fps
    ├── saco_sg
    │   └── JPEGImages_6fps
    └── saco_yt1b
        └── JPEGImages_6fps
```
### Download ready-to-use data
The following links provide ready-to-use data, hosted on Roboflow, after completing the pre-processing steps outlined in the next section.

For each domain:
- [SA-Co/VEval - SA-V](https://universe.roboflow.com/sa-co-veval/sa-v-test/)
- [SA-Co/VEval - YT-Temporal-1B](https://universe.roboflow.com/sa-co-veval/yt-temporal-1b-test/)
- [SA-Co/VEval - SmartGlasses](https://universe.roboflow.com/sa-co-veval/smartglasses-test/)

For all three domains:
- [SA-Co/VEval](https://universe.roboflow.com/sa-co-veval)

Special note on **SA-Co/VEval - YT-Temporal-1B**:
* **Frame Shifting Alert!**
* The ready-to-use data hosted on Roboflow was produced by following the preprocessing steps below. Therefore, the frame-shifting issue for YT-Temporal-1B still exists: due to the nature of Youtube videos, the re-downloaded videos may not be exactly the same as those used during annotation, which can affect eval number reproducibility.

### Download via preprocessing steps
#### Download annotations
The GT annotations are available at Hugging Face:
* [SA-Co/VEval](https://huggingface.co/datasets/facebook/SACo-VEval/tree/main)
    * SA-Co/VEval SA-V
        * Test: `annotation/saco_veval_sav_test.json`
        * Val: `annotation/saco_veval_sav_val.json`
    * SA-Co/VEval YT-Temporal-1B
        * Test: `annotation/saco_veval_yt1b_test.json`
        * Val: `annotation/saco_veval_yt1b_val.json`
    * SA-Co/VEval SmartGlasses
        * Test: `annotation/saco_veval_smartglasses_test.json`
        * Val: `annotation/saco_veval_smartglasses_val.json`

#### Download videos or frames
##### SA-Co/VEval - SAV
Follow instructions in [SA-V dataset](https://ai.meta.com/datasets/segment-anything-video/). Only the following two datasets are needed:
* sav_test.tar
* sav_val.tar

After untar:
```
sav_test/
├── Annotations_6fps [ignore this is the SAM 2 annotation]
├── JPEGImages_24fps
sav_val/
├── Annotations_6fps [ignore this is the SAM 2 annotation]
└── JPEGImages_24fps
```
Then merge the two JPEGImages_24fps together to better match our annotation json file path e.g.
```
media/
    └── saco_sav
        └── JPEGImages_24fps [merged from the two JPEGImages_24fps above]
```
Example commands to download and merge folders
```
cd ../data/media/saco_sav
wget -O sav_test.tar <sav_test.tar download link from the SA-V dataset page>
wget -O sav_val.tar <sav_val.tar download link from the SA-V dataset page>
tar -xf sav_test.tar
tar -xf sav_val.tar
mkdir JPEGImages_24fps
chmod -R u+w sav_test/
chmod -R u+w sav_val/
mv sav_test/JPEGImages_24fps/* JPEGImages_24fps/
mv sav_val/JPEGImages_24fps/* JPEGImages_24fps/
```

##### SA-Co/VEval - YT-Temporal-1B
Two files are needed to download the SA-Co/VEval - YT-Temporal-1B Youtube videos.
* Download `media/yt1b_start_end_time.json` from [SA-Co/VEval](https://huggingface.co/datasets/facebook/SACo-VEval/tree/main), which contains the Youtube video ids and the start and end time used in SA-Co/VEval - YT-Temporal-1B.
* Prepare the `cookies.txt` file. Follow instruction in yt-dlp [exporting-youtube-cookies](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies) and [pass-cookies-to-yt-dlp](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp) to prepare the cookies_file.
    * Please see the full **WARNINGS** in yt-dlp regarding the risk of Youtube account ban!!

Then run `scripts/eval/veval/saco_yt1b_downloader.py` to download the videos and prepare the frames e.g.
```
python saco_yt1b_downloader.py \
--data_dir ../data/media/saco_yt1b \
--cookies_file ../data/media/saco_yt1b/cookies.txt \
--yt1b_start_end_time_file ../data/media/saco_yt1b/yt1b_start_end_time.json \
--yt1b_frame_prep_log_file ../data/media/saco_yt1b/yt1b_frame_prep.log
```
* data_dir: The directoy to download the Youtube videos and store the extraced frames
* cookies_file: the `cookies.txt` downloaded above
* yt1b_start_end_time_file: the `yt1b_start_end_time.json` downloaded above
* yt1b_frame_prep_log_file: a log file to track the video downloading and frame extracting status

Then run `scripts/eval/veval/saco_yt1b_annot_update.py` to update the annotation based on the video availability e.g.
```
python saco_yt1b_annot_update.py \
--yt1b_media_dir ../data/media/saco_yt1b/JPEGImages_6fps \
--yt1b_input_annot_path ../data/annotation/saco_veval_yt1b_val.json \
--yt1b_output_annot_path ../data/annotation/saco_veval_yt1b_val_updated.json \
--yt1b_annot_update_log_path ../data/annotation/saco_veval_yt1b_val_updated.log
```

**NOTE**:
* Not all Youtube videos might be available as Youtube videos can be deleted or become private. The script `saco_yt1b_annot_update.py` is used to remove the annotations of the unavailable videos.
* **Frame Shifting Alert!!** Even when the videos are still available, their specifications, such as fps and duration, may differ from those used during annotation when re-downloaded from YouTube. Additionally, sometimes `ffmpeg` seems to find it hard to guarantee consistent frame extraction from the same video across different environments. This may cause the re-downloaded and re-extracted frames to have alignment issues with our annotations due to frame shifting. Please be aware of this caveat when evaluating on SA-Co/VEval - YT-Temporal-1B.

##### SA-Co/VEval - SmartGlasses
Go to [SACo-VEval](https://huggingface.co/datasets/facebook/SACo-VEval/tree/main) download `media/saco_sg.tar.gz`
```
cd ../data
hf download facebook/SACo-VEval media/saco_sg.tar.gz --repo-type dataset --local-dir .
cd ../data/media
tar -xzf saco_sg.tar.gz
```

## Annotation Format
The format is similar to the [YTVIS](https://youtube-vos.org/dataset/vis/) format.

In the annotation json, e.g. `saco_veval_sav_test.json` there are 5 fields:
* info:
    * A dict containing the dataset info
    * E.g. {'version': 'v1', 'date': '2025-09-24', 'description': 'SA-Co/VEval SA-V Test'}
* videos
    * A list of videos that are used in the current annotation json
    * It contains {id, video_name, file_names, height, width, length}
* annotations
    * A list of **positive** masklets and their related info
    * It contains {id, segmentations, bboxes, areas, iscrowd, video_id, height, width, category_id, noun_phrase}
        * video_id should match to the `videos - id` field above
        * category_id should match to the `categories - id` field below
        * segmentations is a list of [RLE](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py)
* categories
    * A **globally** used noun phrase id map, which is true across all 3 domains.
    * It contains {id, name}
        * name is the noun phrase
* video_np_pairs
    * A list of video-np pairs, including both **positive** and **negative** used in the current annotation json
    * It contains {id, video_id, category_id, noun_phrase, num_masklets}
        * video_id should match the `videos - id` above
        * category_id should match the `categories - id` above
        * when `num_masklets > 0` it is a positive video-np pair, and the presenting masklets can be found in the annotations field
        * when `num_masklets = 0` it is a negative video-np pair, meaning no masklet presenting at all
```
data {
    "info": info
    "videos": [video]
    "annotations": [annotation]
    "categories": [category]
    "video_np_pairs": [video_np_pair]
}
video {
    "id": int
    "video_name": str  # e.g. sav_000000
    "file_names": List[str]
    "height": int
    "width": width
    "length": length
}
annotation {
    "id": int
    "segmentations": List[RLE]
    "bboxes": List[List[int, int, int, int]]
    "areas": List[int]
    "iscrowd": int
    "video_id": str
    "height": int
    "width": int
    "category_id": int
    "noun_phrase": str
}
category {
    "id": int
    "name": str
}
video_np_pair {
    "id": int
    "video_id": str
    "category_id": int
    "noun_phrase": str
    "num_masklets" int
}
```
[sam3/examples/saco_veval_vis_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/saco_veval_vis_example.ipynb) shows some examples of the data format and data visualization.

## Run Offline Eval
An example notebook and an eval script have been provided for offline evaluation.
```
sam3/
├── examples/
│   └── saco_veval_eval_example.ipynb  # this notebook will load eval res or run the eval on the fly, and print the results
└── sam3/eval/
    └── saco_veval_eval.py  # this script will run the offline evaluator
```
`saco_veval_eval.py` supports two modes, `one` and `all`.
* `one`: will take only one pair of gt and pred files to eval
* `all`: will eval on all 6 SACo/VEval datasets

Example usage
```
python saco_veval_eval.py one \
--gt_annot_file ../sam3/assets/veval/toy_gt_and_pred/toy_saco_veval_sav_test_gt.json \
--pred_file ../sam3/assets/veval/toy_gt_and_pred/toy_saco_veval_sav_test_pred.json \
--eval_res_file ../sam3/assets/veval/toy_gt_and_pred/toy_saco_veval_sav_test_eval_res.json
```
* `gt_annot_file`: the location of the GT file
* `pred_file`: the location of the Pred file
* `eval_res_file`: the location where the eval result will be written to

```
python saco_veval_eval.py all \
--gt_annot_dir ../data/annotation \
--pred_dir ../data/pred \
--eval_res_dir ../data/pred
```
* `gt_annot_dir`: the location of the GT files
* `pred_dir`: the location of the Pred files
* `eval_res_dir`: the location where the eval results will be written to
