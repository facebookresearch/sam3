# Download SA-CO Test Data

This document contains information on how to download and setup the image data to prepare them for SA-CO evaluation.

First, download the .json with annotations: (here we will have the URL to download all the .jsons)

(internal [for Shoubhik during testing]): "download" from "/fsx-onevision/didac/data/saco_frames_test_sets/annotations_original" if you want to test the process including removing videos that are no longer on YouTube or Ego4D (at some point users will experience this). Or, "download" from "/fsx-onevision/didac/data/saco_frames_test_sets/annotations" to use the annotations that are already filtered (these will be the ones we will release). I suggest you copy the files to your directory, otherwise the code may modify mine. 


#### >>> TODOs (to synchronize with the team) <<<:

- I added a "image_size" to the ["images"] annotations, to resize if necessary during download. (to make sure the sizes of the frames are the same)
- Update "info" and "licenses" fields in the annotations? (or remove). In general, we need to unify the content of the released jsons.

## Image Datasets

### GeoDE

1. Download dataset with raw images from [GeoDE](https://geodiverse-data-collection.cs.princeton.edu/).
2. Extract the downloaded file to a location, say `<RAW_GEODE_IMAGES_FOLDER>`

3. Run the below command to pre-process the images and prepare for evaluation. The proceesed images will be saved to the location specified in `<PROCESSED_GEODE_IMAGES_FOLDER>`
    ```
    python preprocess_silver_geode_bdd100k_food_rec.py --annotation_file <FOLDER_WITH_SILVER_ANNOTATIONS>/silver_geode_merged_test.json --raw_images_folder <RAW_GEODE_IMAGES_FOLDER> --processed_images_folder <PROCESSED_GEODE_IMAGES_FOLDER> --dataset_name geode
    ```

### National Gallery of Art (NGA)

1. Run the below command to download raw images and pre-process the images to prepare for evaluation. The proceesed images will be saved to the location specified in `<PROCESSED_NGA_IMAGES_FOLDER>`.
    ```
    python download_preprocess_nga.py --annotation_file <FOLDER_WITH_SILVER_ANNOTATIONS>/silver_nga_art_merged_test.json --raw_images_folder <RAW_NGA_IMAGES_FOLDER> --processed_images_folder <PROCESSED_NGA_IMAGES_FOLDER>
    ```

### Berkeley Driving Dataset (BDD) 100k

1. Download data with raw images from the `100K Images` dataset in [BDD100k](http://bdd-data.berkeley.edu/download.html)
2. Extract the downloaded file to a location, say `<RAW_BDD_IMAGES_FOLDER>`
3. Run the below command to pre-process the images and prepare for evaluation. The proceesed images will be saved to the location specified in `<PROCESSED_BDD_IMAGES_FOLDER>`
    ```
    python preprocess_silver_geode_bdd100k_food_rec.py --annotation_file <FOLDER_WITH_SILVER_ANNOTATIONS>/silver_bdd100k_merged_test.json --raw_images_folder <RAW_BDD_IMAGES_FOLDER> --processed_images_folder <PROCESSED_BDD_IMAGES_FOLDER> --dataset_name bdd100k
    ```

### Food Recognition Challenge 2022

1. Download data with raw images from the [website](https://www.aicrowd.com/challenges/food-recognition-benchmark-2022). Download `[Round 2] public_validation_set_2.0.tar.gz` file.
2. Extract the downloaded file to a location, say `<RAW_FOOD_IMAGES_FOLDER>`
3. Run the below command to pre-process the images and prepare for evaluation. The proceesed images will be saved to the location specified in `<PROCESSED_FOOD_IMAGES_FOLDER>`
    ```
    python preprocess_silver_geode_bdd100k_food_rec.py --annotation_file <FOLDER_WITH_SILVER_ANNOTATIONS>/silver_food_rec_merged_test.json --raw_images_folder <RAW_FOOD_IMAGES_FOLDER> --processed_images_folder <PROCESSED_FOOD_IMAGES_FOLDER> --dataset_name food_rec
    ```

### iNaturalist

1. Run the below command to download, extract images in `<RAW_INATURALIST_IMAGES_FOLDER>` and prepare them for evaluation. The proceesed images will be saved to the location specified in `<PROCESSED_INATURALIST_IMAGES_FOLDER>`
    ```
    python download_inaturalist.py --raw_images_folder <RAW_INATURALIST_IMAGES_FOLDER> --processed_images_folder <PROCESSED_INATURALIST_IMAGES_FOLDER>
    ```

### Fathomnet

1. Install the FathomNet API
    ```
    pip install fathomnet
    ```

2. Run the below command to download the images and prepare for evaluation. The proceesed images will be saved to the location specified in `<PROCESSED_BDD_IMAGES_FOLDER>`
    ```
    python download_fathomnet.py --processed_images_folder <PROCESSED_BFATHOMNET_IMAGES_FOLDER>
    ```

## Frame Datasets

These datasets correspond to annotations for individual frames coming from videos. The file `CONFIG_FRAMES.yaml` is used to unify the downloads for the datasets, as explained below.

Before following the other dataset steps, update `CONFIG_FRAMES.yaml` with the correct `path_annotations` path where the annotation files are.

### DROID

1. Install the gsutil package:
    ```bash
    pip install gsutil
    ```
2. Modify the `droid_path` variable in `CONFIG_FRAMES.yaml`. This is the path where the DROID data will be downloaded.
3. _\[Optional\] Update the variable `remove_downloaded_videos_droid` to (not) remove the videos after the frames have been extracted.
4. Download the data:
    ```bash
    python download_videos.py droid
    ```
5. Extract the frames:
    ```bash
    python extract_frames.py droid
    ```

See the [DROID website](https://droid-dataset.github.io/droid/the-droid-dataset#-using-the-dataset) for more information.

### SA-V

1. Follow instructions in the [Segment Anything official website](https://ai.meta.com/datasets/segment-anything-video-downloads/) to obtain access to the download links (they are dynamic links).
2. Update `CONFIG_FRAMES.yaml`:
    - Update the `sav_path` variable, where the frames will be saved.
    - Update the `sav_videos_fps_6_download_path` variable. Copy paste the path corresponding to the `videos_fps_6.tar` in the list that you obtained in step 1.
    - _\[Optional\]_ Update the variable `remove_downloaded_videos_sav` to (not) remove the videos after the frames have been extracted.
3. Download the videos:
    ```bash
    python download_videos.py sav
    ```
4. Extract the frames: 
    ```
    python extract_frames.py sav
    ```

### Ego4D

1. Review and accept the license agreement in the [official Ego4D website](https://ego4d-data.org/docs/start-here/#license-agreement).
2. Configure AWS credentials. Run:
    ```bash
    pip install awscli
    aws configure
    ```
    and copy the values shown in the email you received after step 1 (you can leave "region name" and "output format" empty). You can verify that the variables were set up correctly:
    ```bash
    cat ~/.aws/credentials
    ```
3. Install the Ego4D library: 
    ```bash
    pip install ego4d
    ```
4. Update `CONFIG_FRAMES.yaml`:
    - Set up AWS credentials following the instructions in the email you received after step 2. Modify the following variables: `aws_access_key_id` and `aws_secret_access_key`.
    - Update the `ego4d_path` variable, where the frames will be saved.
    - _\[Optional\]_ Update the variable `remove_downloaded_videos_ego4d` to (not) remove the videos after the frames have been extracted..
5. Download the `clips` subset of the Ego4D dataset:
    ```python
    python download_videos.py ego4d
    ```
6. Extract the frames: 
    ```
    python extract_frames.py ego4d
    ```

See the [official CLI](https://ego4d-data.org/docs/CLI/) and the [explanation about the videos](https://ego4d-data.org/docs/data/videos/) for more information.

### YT1B
1. Install the yt-dlp library:
    ```bash
    python3 -m pip install -U "yt-dlp[default]"
    ```
2. Create a `cookies.txt` file following the instructions from yt-dlp [exporting-youtube-cookies](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies) and [pass-cookies-to-yt-dlp](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp). This is required to download youtube videos. Then, update the path for that file in the `CONFIG_FRAMES.yaml` file, in the variable `cookies_path`.
3. Update `CONFIG_FRAMES.yaml`:
    - Update the `yt1b_path`, where the frames will be saved.
    - _\[Optional\]_ Some YouTube videos may not be available on YouTube anymore. Set `update_annotation_yt1b` to `True` in `CONFIG_FRAMES.yaml` to remove the annotations corresponding to such videos. Note that the evaluations will not be directly comparable with other reported evaluations. 
    - _\[Optional\]_ Update the variable `remove_downloaded_videos_yt1b` to (not) remove the videos after the frames have been extracted.
4. Run the following code to download the videos:
    ```
    python download_videos.py yt1b
    ```
5. Extract the frames: 
    ```
    python extract_frames.py yt1b
    ```
