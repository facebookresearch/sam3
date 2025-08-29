# Evaluation dataset

SALLY-Gold is an evaluation dataset comprising of 7 subsets, each targeting a different scenario. For each subset, the annotations are multi-reviewed and agreed by 3 human annotators resulting in a high quality test set.

## Usage

### Download annotations

The annotations can be downloaded from the following location: https://drive.google.com/drive/folders/1o4IKrf3BvFMmUabpgZdZVDB5b7xG0CWB?usp=drive_link

### Download images

There are two image sources for the evaluation dataset: MetaCLIP and SA-1B.

1) The MetaCLIP images are referred in 6 out of 7 subsets and can be downloaded using the below commond. This will download all the images from the urls referred in `gold_metaclip_filename_urls_mapping.json`. The script should download 22507 images.

```bash
python download_metaclip_urls_submitit.py
```

2) The SA-1B images can be referred from the publicly released version.

### Update annotations (Optional)

If the number of MetaCLIP images downloaded is less than 22507, then run the below script to update the groundtruth annotations. This will filter annotations related to missing images.
```bash
python update_eval_files.py
```

### Run evaluation

Run the below command for offline evaluation.

```bash
python eval_sam3.py
```
