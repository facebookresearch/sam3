# SA-Co/VEval and SA-FARI Dataset
**SA-Co/VEval** is an evaluation dataset comprising of 3 domains and 6 subests, each domain has a val and test.
* SA-Co/VEval - SAV: videos are from the [SA-V dataset](https://ai.meta.com/datasets/segment-anything-video/)
* SA-Co/VEval - YT1B: videos are from the [YT-Temporal-1B](https://cove.thecvf.com/datasets/704)
* SA-Co/VEval - SmartGlasses: egocentric videos from smart glasses

**SA-FARI** is an evaluation dataset comprising 1 domain and 1 subset.
* SA-FARI: videos are from wildlife cameras, partnership with [Conservation X Labs](https://www.conservationxlabs.com/)

## Usage
### Download annotations
The GT annotations can be downloaded from the following [location](https://drive.google.com/drive/folders/1BadVFUfENo5JsehDWKuYbTllS20JtmiX) [TODO: update to HF]

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
Run `saco_yt1b_downloader.py` to download youtube videos used in the SA-Co/VEVal - YT1B dataset.
```
python saco_yt1b_downloader.py \
--data_dir /fsx-onevision/tym/sam3_and_data/data/media/saco_yt1b \
--cookies_file /fsx-onevision/tym/sam3_and_data/data/media/saco_yt1b/cookies.txt \
--id_map_file /fsx-onevision/tym/sam3_and_data/data/media/saco_yt1b/id_and_frame_map.json \
--download_result /fsx-onevision/tym/sam3_and_data/data/media/saco_yt1b/download_result.txt
```
* data_dir: The directoy where to store the downloaded youtube videos
* cookies_file: This is required to download youtube videos. See instructions from yt-dlp [exporting-youtube-cookies](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies) and [pass-cookies-to-yt-dlp](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp) to prepare the cookies_file
* id_map_file: download from [location](https://drive.google.com/drive/folders/1BadVFUfENo5JsehDWKuYbTllS20JtmiX) `id_and_frame_map.json`
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
[TODO: finish writing here]
