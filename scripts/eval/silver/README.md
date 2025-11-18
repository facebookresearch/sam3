# SA-Co/Silver benchmark

SA-Co/Silver is a benchmark for promptable concept segmentation (PCS) in images. The benchmark contains images paired with text labels (also referred as Noun Phrases aka NPs), each annotated exhaustively with masks on all object instances that match the label.

SA-Co/Silver comprises 10 subsets, covering a diverse array of domains including food, art, robotics, driving etc.

- BDD100k
- DROID
- Ego4D
- MyFoodRepo-273
- GeoDE
- iNaturalist-2017
- National Gallery of Art
- SA-V
- YT-Temporal-1B
- Fathomnet

The README contains instructions on how to download and setup the annotations, image data to prepare them for evaluation on SA-Co/Silver.

## Download annotations

The GT annotations can be downloaded from [Hugging Face](https://huggingface.co/datasets/facebook/SACo-Silver) or [Roboflow](https://universe.roboflow.com/sa-co-silver)

## Download images and video frames

### Image Datasets

#### GeoDE

The processed images needed for evaluation can be downloaded from [Roboflow](https://universe.roboflow.com/sa-co-silver/geode/) OR follow the below steps to prepare the processed images.

1. Download dataset with raw images from [GeoDE](https://geodiverse-data-collection.cs.princeton.edu/).
2. Extract the downloaded file to a location, say `<RAW_GEODE_IMAGES_FOLDER>`

3. Run the below command to pre-process the images and prepare for evaluation. The proceesed images will be saved to the location specified in `<PROCESSED_GEODE_IMAGES_FOLDER>`
    ```
    python preprocess_silver_geode_bdd100k_food_rec.py --annotation_file <FOLDER_WITH_SILVER_ANNOTATIONS>/silver_geode_merged_test.json --raw_images_folder <RAW_GEODE_IMAGES_FOLDER> --processed_images_folder <PROCESSED_GEODE_IMAGES_FOLDER> --dataset_name geode
    ```

#### National Gallery of Art (NGA)

The processed images needed for evaluation can be downloaded from [Roboflow](https://universe.roboflow.com/sa-co-silver/national-gallery-of-art/) OR follow the below steps to prepare the processed images.

1. Run the below command to download raw images and pre-process the images to prepare for evaluation. The proceesed images will be saved to the location specified in `<PROCESSED_NGA_IMAGES_FOLDER>`.
    ```
    python download_preprocess_nga.py --annotation_file <FOLDER_WITH_SILVER_ANNOTATIONS>/silver_nga_art_merged_test.json --raw_images_folder <RAW_NGA_IMAGES_FOLDER> --processed_images_folder <PROCESSED_NGA_IMAGES_FOLDER>
    ```

#### Berkeley Driving Dataset (BDD) 100k

The processed images needed for evaluation can be downloaded from [Roboflow](https://universe.roboflow.com/sa-co-silver/bdd100k-gwmh6/) OR follow the below steps to prepare the processed images.

1. Download data with raw images from the `100K Images` dataset in [BDD100k](http://bdd-data.berkeley.edu/download.html)
2. Extract the downloaded file to a location, say `<RAW_BDD_IMAGES_FOLDER>`
3. Run the below command to pre-process the images and prepare for evaluation. The proceesed images will be saved to the location specified in `<PROCESSED_BDD_IMAGES_FOLDER>`
    ```
    python preprocess_silver_geode_bdd100k_food_rec.py --annotation_file <FOLDER_WITH_SILVER_ANNOTATIONS>/silver_bdd100k_merged_test.json --raw_images_folder <RAW_BDD_IMAGES_FOLDER> --processed_images_folder <PROCESSED_BDD_IMAGES_FOLDER> --dataset_name bdd100k
    ```

#### Food Recognition Challenge 2022

1. Download data with raw images from the [website](https://www.aicrowd.com/challenges/food-recognition-benchmark-2022). Download `[Round 2] public_validation_set_2.0.tar.gz` file.
2. Extract the downloaded file to a location, say `<RAW_FOOD_IMAGES_FOLDER>`
3. Run the below command to pre-process the images and prepare for evaluation. The proceesed images will be saved to the location specified in `<PROCESSED_FOOD_IMAGES_FOLDER>`
    ```
    python preprocess_silver_geode_bdd100k_food_rec.py --annotation_file <FOLDER_WITH_SILVER_ANNOTATIONS>/silver_food_rec_merged_test.json --raw_images_folder <RAW_FOOD_IMAGES_FOLDER> --processed_images_folder <PROCESSED_FOOD_IMAGES_FOLDER> --dataset_name food_rec
    ```

#### iNaturalist

The processed images needed for evaluation can be downloaded from [Roboflow](https://universe.roboflow.com/sa-co-silver/inaturalist-2017/) OR follow the below steps to prepare the processed images.

1. Run the below command to download, extract images in `<RAW_INATURALIST_IMAGES_FOLDER>` and prepare them for evaluation. The proceesed images will be saved to the location specified in `<PROCESSED_INATURALIST_IMAGES_FOLDER>`
    ```
    python download_inaturalist.py --raw_images_folder <RAW_INATURALIST_IMAGES_FOLDER> --processed_images_folder <PROCESSED_INATURALIST_IMAGES_FOLDER>
    ```

#### Fathomnet

The processed images needed for evaluation can be downloaded from [Roboflow](https://universe.roboflow.com/sa-co-silver/fathomnet-kmz5d/) OR follow the below steps to prepare the processed images.

1. Install the FathomNet API
    ```
    pip install fathomnet
    ```

2. Run the below command to download the images and prepare for evaluation. The proceesed images will be saved to the location specified in `<PROCESSED_BDD_IMAGES_FOLDER>`
    ```
    python download_fathomnet.py --processed_images_folder <PROCESSED_BFATHOMNET_IMAGES_FOLDER>
    ```

### Frame Datasets

These datasets correspond to annotations for individual frames coming from videos. The file `CONFIG_FRAMES.yaml` is used to unify the downloads for the datasets, as explained below.

Before following the other dataset steps, update `CONFIG_FRAMES.yaml` with the correct `path_annotations` path where the annotation files are.

#### DROID

The processed frames needed for evaluation can be downloaded from [Roboflow](https://universe.roboflow.com/sa-co-silver/droid-cfual/) OR follow the below steps to prepare the processed frames.

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

#### SA-V

The processed frames needed for evaluation can be downloaded from [Roboflow](https://universe.roboflow.com/sa-co-silver/sa-v) OR follow the below steps to prepare the processed frames.

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

#### Ego4D

The processed frames needed for evaluation can be downloaded from [Roboflow](https://universe.roboflow.com/sa-co-silver/ego4d-w7fiu/) OR follow the below steps to prepare the processed frames.

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

#### YT1B

The processed frames needed for evaluation can be downloaded from [Roboflow](https://universe.roboflow.com/sa-co-silver/yt-temporal-1b/) OR follow the below steps to prepare the processed frames.

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

## Visualization

- Visualize GT annotations: [saco_gold_silver_vis_example.ipynb](https://github.com/facebookresearch/sam3/blob/main/examples/saco_gold_silver_vis_example.ipynb)

## Run online evaluation

Update the path for GT annotation and images and run the below command for online evaluation.

```bash
python sam3/train/train.py -c configs/silver_image_evals/sam3_silver_image_bdd100k.yaml --use-cluster 1
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

A sample annotation from DROID domain looks as follows:

#### images

```
[
  {
    "id": 10000000,
    "file_name": "AUTOLab_failure_2023-07-07_Fri_Jul__7_18:50:36_2023_recordings_MP4_22008760/00002.jpg",
    "text_input": "the large wooden table",
    "width": 1280,
    "height": 720,
    "queried_category": "3",
    "is_instance_exhaustive": 1,
    "is_pixel_exhaustive": 1
  }
]
```

#### annotations

```
[
  {
    "area": 0.17324327256944444,
    "id": 1,
    "image_id": 10000000,
    "source": "created by SAM3",
    "bbox": [
      0.03750000149011612,
      0.5083333253860474,
      0.8382812738418579,
      0.49166667461395264
    ],
    "segmentation": {
      "counts": "[^R11]f03O0O100O2N100O1O100O100O100O100O1O100O100O100O100O100O1O10000O1O10000O1O100O10000O1O100O100O100O100O100O100O100O100O100O100O1O100O100O10000O100O100O100O101N100O1O011O0O1O101OO0010O100O1O100O2OO0100O100O100O100O100O10000O100O100O1O100O10000O1O100O100O100O10000O1O100O100O100O10000O1O10000O1O100O100O100O100O100O100O1O100O100O100O100O100O100O100O100O100O100O100O100O100O100O10000O100O100O1O100O10000O100O100O100O100O1O100O100O100O100O100O100O10O0100O100O2O000O1O10000O1O10000O100O100O100O1O100O100O100O100O100O100O100O100O100O100O100O100O1O100O100O100O10000O100O100O100O100O100O100O100O100O100O100O100O100O100O10000O100O100O100O100O100O100O1O10000O1O10000O100O1O100O100O100O100O100O100O100O100O10000O1O100O100O100O100O1O10000O10\\MP@hNo?W1U@gNk?X1W@gNh?Y1Z@fNf?Y1\\@fNc?[1^@dNb?[1`@dN_?]1b@bN^?]1e@aNZ?_1i@_NW?a1l@\\NS?d1RAXNn>h1TAVNk>k1VATNj>k1XATNg>m1YASNg>m1YASNf>m1[ASNe>m1[ASNd>m1]ASNc>m1]ASNb>l1`ATN`>i1cAWN\\>d1jA\\NV>_1oAaNP>^1RBbNn=\\1TBdNk=\\1VBdNj=1`@dNGO02P2Z1h=L_AfNj0^1g=FmC;R<EoC;Q<DPD<o;DRD<n;DQD=n;DjAnN?^1g=DhAQO?\\1h=DhAUO<W1l=EeAZO:R1P>F]ABa0h0Q>Hd@lNDV1e17S>k1iAWNW>i1hAXNW>j1gAWNY>i1fAXNY>j1eAWNZ>k1dAVN\\>k1bAVN^>k1`AVN_>l1`ATN`>m1^ATNa>o1]AQNc>P2[AQNd>P2\\APNd>Q2[AoMd>R2[AoMd>R2\\AnMd>S2ZAnMe>S2[AmMe>T2YAmMf>T2YAmMg>T2WAmMh>U2VAlMj>U2TAlMl>U2PAnMo>U2j@PNV?e4O100O100O100O100O100O100O100O100O100O100O100O100O101N100O100O10O0100O100O100O100O100O100O1000000O1000000O100O100O1O1O1O100O100O1O100O100O100O100O100O100O100O100O100O1O100O100O100O100O100O10000O100O1O100O100O100O100O100O100OkK_B]Oa=7oBEP=4YCKg<1^CNa<1bCN^<OeC1[<LhC4W<KlC4S<KoC5Q<JPD6o;JRD6n;JSD5l;LTD4l;LTD4k;MUD3k;MUD4j;LWD2i;OWD1i;OWD1h;0XD0h;1WDOh;2XDOg;1ZDNe;3[DMe;3[DNc;3]DLd;4\\DLc;5]DKb;7]DIc;7^DHa;9_DGa;9_DG`;:`DF`;;_DE`;<`DCa;=^DDa;=_DC`;>_DCa;>^DBb;[OUCiMW1n2c;YO[CeMn0V3g;TO^CeMf0[3k;POaCdM>b3Q<iNbCfM7f3V<dNeCeMKQ4`<YNgCfMAX4g<RNiCk2W<SMlCl2S<TMnCl2R<SMoCm2Q<RMQDm2n;TMRDl2n;SMTDl2k;UMUDk2k;UMVDj2i;VMXDj2h;VMXDj2g;VM[Di2e;VM\\Dj2c;VM^Dj2b;TMaDk2^;PMhDP3X;aL`CjM`1e5o:\\L^Ed3b:WLdEh3[:nKPFR4P:jKTFV4k9hKXFX4h9hKXFX4g9hKYFY4f9hKZFX4f9hKZFX4e9iKZFW4g9iKXFX4g9iKPElN\\O\\5c;iKeDYOEo4f;iK]DAJh4g;iKTDJ3^4i;jKkCO;X4i;hMVDX2j;hMUDY2j;iMUDW2k;iMTDW2l;kMSDU2m;kMRDV2m;lMRDT2n;mMPDT2P<mMoCS2P<oMnCR2R<V4O100O100OiInCR2Q<kMWDQ2i;kM_DQ2`;lMoDi1Q;TNWEg1h:XN^Ed1a:\\NdE`1\\:^NjE^1U:aNPF]1o9aNUF]1k9bNXF\\1g9dN]FY1c9fN`FX1_9hNdFV1\\9iNhFT1W9lNmFQ1S9nNQGo0n8QOTGn0l8ROWGk0h8UO[Gi0e8VO^Gh0a8YO`Gf0`8YOcGe0\\8\\OeGc0[8\\OiGa0V8@lG>T8AnG>Q8BQH=o7CRH<m7DVH:j7FWH9h7HYH7g7H[H7d7J^H4b7L^H4b7K`H4_7MbH2^7NcH1\\7OfH0Z70gHOX72iHMW73jHLV74jHLU74mHKS75mHKS75nHJR76oHIQ77oHIR7jMkDP1U4U1S7RM_D0h0g1f3W1^8hNcGV1_8iNaGX1_8gNaGY1`8fNaGY1_8gNaGY1`8fNaGY1_8gNaGY1`8fNaGY1_8gNaGY1`8fNaGY1_8gNaGY1_8gNaGY1_8gNbGX1_8gNaGY1_8gNaGY1_8fNbGY1`8fNaGY1_8gNaGY1_8gNaGY1_8gNaGY1_8gNbGX1^8hNbGX1^8hNbGX1^8hNbGX1^8hNbGX1^8iNbGV1^8jNbGV1^8jNbGV1^8jNbGV1^8jNbGV1^8jNbGV1^8jNbGV1]8lNbGT1^8lNcGS1\\8nNdGR1\\8nNdGR1[8oNeGQ1Z8POfGP1X8SOhGl0W8UOiGk0U8WOkGi0S8YOmGg0P8\\OPHd0n7_ORH`0l7BTH>j7DVH<g7HYH7d7L\\H4b7N^H2`71_HO^74bHL[77eHIY7:fHFX7<hHDV7>jHBT7a0kH_OT7b0mH]OR7d0nH\\OQ7f0nH]OQ7g0oHZOQ7g0oHYOQ7h0nHXOR7h0nHXOR7h0nHXOR7i0mHWOT7h0kHYOU7h0jHXOV7h0iHYOW7g0iHYOW7h0hHXOY7g0fHZOZ7f0eH[O\\7e0cHhNlKSNa;U3bHeNSLTN\\;W3_HbN]LRNU;\\3]H^Nb8c1\\G\\Ng8c1XG\\Nj8e1TGZNo8e1PGYNS9h1lFUNW9l1gFRN]9m1bFRN`9o1^FPNe9o1[FoMg9R2WFnMj9S2TFmMn9R2RFnMn9S2PFmMR:R2nEmMS:T2kEmMU:T2jEkMX:T2gEmMY:T2fElMZ:U2dEkM^:T2aEmM_:T2`ElM`:U2^ElMc:S2\\EmMe:T2YEmMg:T2WEmMj:S2UEmMk:T2SEmMn:S2PEnMP;S2nDoMQ;R2mDoMT;Q2kDoMU;R2iDoMX;Q2fDQNY;P2eDQN[;P2cDQN^;o1`DSN_;n1^DTNc;l1[DVNd;k1ZDVNg;j1WDXNh;j1UDWNk;j1SDWNn;i1oCZNP<h1mCYNS<h1kCZNU<g1gC\\NX<e1fC\\N[<d1cC^N\\<d1aC^N_<c1^C_Na<b1\\CaNc<a1ZCaNf<_1XCcNg<_1UCeNj<^1oBfNP=]1iBiN?gL^;e4hCkNf0dLb;`8YDcGg;^8VDdGk;^8mChGR<_8bCfG_<U900001N101O00001O001O00001O00001O0O2N1O1O2N1O2N100O2N1O1O2N1O2N1O1O2N1O2M200O2M2O2N1N2O2N1N3N1O1N3N1N3M2O2kMkAkKW>Q4RBiKo=8^AR2j0`Mk=:aAP2i0bMh==eAj1g0eMf=?hAh1f0eMd=?lAg1c0gMc=`0nAe1c0hMa=a0oAd1b0iM`=a0QBc1c0iM]=c0SB`1d0iM\\=e0SB^1e0jMY=g0VB[1e0jMV=k0WBW1V`0gNn_OT1T`0lNo_Oo0S`0POS@i0P`0VOT@d0n?\\OT@`0n?@T@<o?CR@^OUN6ka0=P@XO\\N6ga0a0j@WOY?i0X3O001O00010O00001O0010O0001O00010O001O00001O001O01O01O00001O001O000O2O0O2O0O2N1O2N1O2M3MYl51fSJ3L3O1O100O1O100000000001O000000001O00000000001O01OO1000000000001O000001O000O10000000000000000O10000O10000O10000O100O1O100O1O1O1O1O1O1N2O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1N2O1O1O1O1O1O1O100O100N21O00001O001O2N1O1O2N1O2N1O2M3N4IVT_3",
      "size": [
        720,
        1280
      ]
    },
    "category_id": 1,
    "iscrowd": 0
  }
]
```

### Data Stats

Here are the stats for the 10 annotation domains. The # Image-NPs represent the total number of unique image-NP pairs including both “positive” and “negative” NPs. 


| Domain                   | # Image-NPs  | # Image-NP-Masks|
|--------------------------|--------------| ----------------|
| BDD100k                  | 5546         | 13210           |
| DROID                    | 9445         | 11098           |
| Ego4D                    | 12608        | 24049            |
| MyFoodRepo-273           | 20985        | 28347           |
| GeoDE                    | 14850        | 7570            |
| iNaturalist-2017         | 1439051      | 48899           |
| National Gallery of Art  | 22294        | 18991            |
| SA-V                     | 18337        | 39683            |
| YT-Temporal-1B           | 7816         | 12221            |
| Fathomnet                | 287193         | 14174            |
