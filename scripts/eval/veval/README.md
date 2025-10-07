# SA-Co/VEval and SA-FARI Dataset
**SA-Co/VEval** is an evaluation dataset comprising of 3 domains and 6 subests, each domain has a val and test.
* SA-Co/VEval - SAV: videos are from the [SA-V dataset](https://ai.meta.com/datasets/segment-anything-video/)
* SA-Co/VEval - YT1B: videos are from the [YT-Temporal-1B](https://cove.thecvf.com/datasets/704)
* SA-Co/VEval - SmartGlasses: egocentric videos from smart glasses

**SA-FARI** is an evaluation dataset comprising 1 domain and 1 subset.
* SA-FARI: videos are from wildlife cameras, partnership with [Conservation X Labs](https://www.conservationxlabs.com/)

## Usage
Install the SA-Co/VEVal required environment
```
pip install -e ".[veval]"
```
### Download annotations
The GT annotations can be downloaded from the following [location](https://drive.google.com/drive/folders/1BadVFUfENo5JsehDWKuYbTllS20JtmiX) folder `gt`
[TODO: update to HF]
[TODO: update the GDrive files gt and pred to the latest launched ver.]

### Download videos or frames
#### SA-Co/VEval - SAV
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
We recommend to merge the two JPEGImages_24fps together e.g.
```
media/
    └── saco_sav
        └── JPEGImages_24fps [merged from the two JPEGImages_24fps above]
```
#### SA-Co/VEval - YT1B
[TODO: provide the latest yt1b_id_frame_map to download that fixes the yt1b frame matching]
Run `saco_yt1b_downloader.py` to download youtube videos used in the SA-Co/VEVal - YT1B dataset.
```
python saco_yt1b_downloader.py \
--data_dir /fsx-onevision/tym/sam3_and_data/data/media/saco_yt1b \
--cookies_file /fsx-onevision/tym/sam3_and_data/data/media/saco_yt1b/cookies.txt \
--id_map_file /fsx-onevision/tym/sam3_and_data/data/media/saco_yt1b/yt1b_id_frame_map.json \
--yt1b_frame_prep_log_path /fsx-onevision/tym/sam3_and_data/data/media/saco_yt1b/yt1b_frame_prep_log.log
```
* data_dir: The directoy where to store the downloaded youtube videos
* cookies_file: This is required to download youtube videos. See instructions from yt-dlp [exporting-youtube-cookies](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies) and [pass-cookies-to-yt-dlp](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp) to prepare the cookies_file
* id_map_file: download from [location](https://drive.google.com/drive/folders/1BadVFUfENo5JsehDWKuYbTllS20JtmiX) `yt1b_id_frame_map.json`
* download_result: a log file to track if the youtube videos are still downloadable or not
#### SA-Co/VEval - SmartGlasses
[TODO: HF setup]

#### SA-Co/VEval - SA-FARI
[TODO: CXL setup]

### The folder structure
The following folder structure is expected after finishing all the downloads and pre-processing:
```
data/
├── annotation/
│   ├── sa_fari_test.json
│   ├── saco_veval_sav_test.json
│   ├── saco_veval_sav_val.json
│   ├── saco_veval_smartglasses_test.json
│   ├── saco_veval_smartglasses_val.json
│   ├── saco_veval_yt1b_test.json
│   ├── saco_veval_yt1b_val.json
└── media/
    ├── sa_fari
    │   └── JPEGImages_6fps
    ├── saco_sav
    │   └── JPEGImages_24fps
    ├── saco_sg
    │   └── JPEGImages_6fps
    └── sa_yt1b
        └── JPEGImages_6fps
```
## Annotation Format
The format is similar to the [YTVIS](https://youtube-vos.org/dataset/vis/) format.

If we load a json, e.g. `saco_veval_sav_test.json` it will have 5 fields:
* info:
    * A dict containing the dataset info
    * E.g. {'version': 'v1', 'date': '2025-09-24', 'description': 'SA-Co/VEval SA-V Test'}
* videos
    * A list of videos that are used in the current annotation json
    * It contains {id, file_names, height, width, length}
* annotations
    * A list of **positive** masklets and their related info
    * It contains {id, segmentations, bboxes, areas, iscrowd, video_id, height, width, category_id, noun_phrase}
        * video_id should match the `videos - id` above
        * category_id should match the `categories - id` below
        * segmentations is a list of [RLE](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py)
* categories
    * A list of noun phrases that are **globally** used, e.g. not only containing the noun phrases are used in the current annotation json but also covering all the others.
        * That being said, all 7 datasets in SA-Co/VEval and SA-FARI share the exact same categories.
    * It contains {id, name}
        * name is the noun phrase
* video_np_pairs
    * A list of video-np pairs, including both **positive** and **negative** used in the current annotation json
    * It contains {id, video_id, category_id, noun_phrase}
        * video_id should match the `videos - id` above
        * category_id should match the `categories - id` above

```
data {
    "info": info
    "videos": [video]
    "annotations": [annotation]
    "categories": [category]
    "video_np_pairs": [video_np_pair]
}
video {
    "id": str  # e.g. sav_000000
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
}
```
In `veval/saco_veval_example.ipynb` we can find more concrete examples.

## Run Eval
The example pred annotations can be downloaded from the following [location](https://drive.google.com/drive/folders/1BadVFUfENo5JsehDWKuYbTllS20JtmiX) folder `pred`
```
python saco_veval_eval.py \
--gt_ann_file /fsx-onevision/tym/sam3_and_data/data/annotation/saco_veval_sav_test.json \
--pred_file /fsx-onevision/tym/sam3_and_data/data/pred/example_09242025/saco_sav_test_preds.json
```
The `saco_veval_eval.py` will run
* VideoTetaEvaluator
* VideoPhraseHotaEvaluator
* VideoDemoF1Evaluator
The results will be available in the same folder of `--pred_file` with a suffix `_res` e.g. `--pred_file /fsx onevision/tym/sam3_and_data/data/pred/example_09242025/saco_sav_test_preds_res.json`

For a toy run without the actual downloads needed, directly run
```
python saco_veval_eval.py
```
It will use the toy data in `veval/toy_gt_and_pred` and generate an eval result in `veval/toy_gt_and_pred/toy_saco_sav_test_preds_res.json`
