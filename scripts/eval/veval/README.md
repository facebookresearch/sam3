python saco_yt1b_frame_prep.py --saco_yt1b_id saco_yt1b_000403 --data_dir /fsx-onevision/tym/sam3_and_data/data/media/saco_yt1b --cookies_file /fsx-onevision/tym/sam3_and_data/data/media/saco_yt1b/cookies.txt --id_map_file /fsx-onevision/tym/sam3_and_data/data/media/saco_yt1b/id_and_frame_map.json

python saco_yt1b_frame_prep.py --saco_yt1b_id saco_yt1b_000189 --data_dir /fsx-onevision/tym/sam3_and_data/data/media/saco_yt1b --cookies_file /fsx-onevision/tym/sam3_and_data/data/media/saco_yt1b/cookies.txt --id_map_file /fsx-onevision/tym/sam3_and_data/data/media/saco_yt1b/id_and_frame_map.json


tar -xf sav_test.tar -C .

mv /fsx-onevision/tym/sam3_and_data/data/media/saco_sav/sav_val/JPEGImages_24fps /fsx-onevision/tym/sam3_and_data/data/media/saco_sav/JPEGImages_24fps


chmod -R u+w sav_test/JPEGImages_24fps/
chmod -R u+w sav_val/JPEGImages_24fps/

mv the jpegimages to the following structure (so vis code can properly locate them)

the ending media folder strcuture will be:
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


they all shared the same categories

video np pair contains both positives and negatives
where annotations only has postivies with masklets