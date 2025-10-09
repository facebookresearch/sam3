# SA-Co/VEval Dataset
**SA-Co/VEval** is an evaluation dataset comprising of 3 domains, each domain has a val and test split.
* SA-Co/VEval - SA-V: videos are from the [SA-V dataset](https://ai.meta.com/datasets/segment-anything-video/)
* SA-Co/VEval - YT-Temporal-1B: videos are from the [YT-Temporal-1B](https://cove.thecvf.com/datasets/704)
* SA-Co/VEval - SmartGlasses: egocentric videos from smart glasses

## Environment
Install the SA-Co/VEVal required environment
```
pip install -e ".[veval]"
```
This will allow us to run:
* `scripts/eval/veval/saco_yt1b_downloader.py` preparing frames for SA-Co/VEval - YT1B
* `scripts/eval/veval/saco_veval_eval.py` example of running an offline evaluator
* `examples/saco_veavl_example.ipynb` example of loading and visualizing the data

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
    └── sa_yt1b
        └── JPEGImages_6fps
```

### Download annotations
The GT annotations are available at Hugging Face:
* [SACo-VEval](https://huggingface.co/datasets/facebook/SACo-VEval/tree/main) (HF repo is private now. Another temp [GDrive](https://drive.google.com/drive/folders/1p26lWYxW1p0ElNBLe6KiVQiv4_fyp6hO) location for sharing.)
    * SA-Co/VEval SA-V
        * Test: `annotation/saco_veval_sav_test.json`
        * Val: `annotation/saco_veval_sav_val.json`
    * SA-Co/VEval YT-Temporal-1B
        * Test: `annotation/saco_veval_yt1b_test.json`
        * Val: `annotation/saco_veval_yt1b_val.json`
    * SA-Co/VEval SmartGlasses
        * Test: `annotation/saco_veval_smartglasses_test.json`
        * Val: `annotation/saco_veval_smartglasses_val.json`

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

#### SA-Co/VEval - YT-Temporal-1B
Two files needed to download the SA-Co/VEval - YT-Temporal-1B Youtube videos.
* Go to [SACo-VEval](https://huggingface.co/datasets/facebook/SACo-VEval/tree/main) download `media/yt1b_id_frame_map.json`, which contains which Youtube videos and which frames are used for the SA-Co/VEval - YT-Temporal-1B.
* Prepare the `cookies.txt` file. Follow instruction in yt-dlp [exporting-youtube-cookies](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies) and [pass-cookies-to-yt-dlp](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp) to prepare the cookies_file.
    * Please see the full WARNINGS in yt-dlp regarding the risk of Youtube account ban!!

Then run `scripts/eval/veval/saco_yt1b_downloader.py` e.g.
```
python saco_yt1b_downloader.py \
--data_dir ../data/media/saco_yt1b \
--cookies_file ../data/media/saco_yt1b/cookies.txt \
--id_map_file ../data/media/saco_yt1b/yt1b_id_frame_map.json \
--yt1b_frame_prep_log_path ../data/media/saco_yt1b/yt1b_frame_prep_log.log
```
* data_dir: The directoy to download the Youtube videos and store the extraced frames
* cookies_file: the `cookies.txt` downloaded above
* id_map_file: the `yt1b_id_frame_map.json` downloaded above
* yt1b_frame_prep_log_path: a log file to track the downloader status, including the stages of video download, frame extracting, and frame matching, for each video.

Note: not all Youtube videos might be available since some videos might be deleted or moved from public to private.
  
#### SA-Co/VEval - SmartGlasses
Go to [SACo-VEval](https://huggingface.co/datasets/facebook/SACo-VEval/tree/main) download `media/saco_sg.tar.gz` (HF repo is private now. Another temp [GDrive]([https://drive.google.com/drive/folders/1p26lWYxW1p0ElNBLe6KiVQiv4_fyp6hO](https://drive.google.com/drive/folders/1aitfOfBfelJZNQGbRHgw00bxNiZlyVSM)) location `saco_sg.tag` for sharing.)
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
`veval/saco_veval_example.ipynb` has more examples to show the annotation structure.
