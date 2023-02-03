# Downloading the MUSIC Dataset

This folder contains the scripts for downloading the MUSIC dataset. The JSON files are downloaded from the original [repository](https://github.com/roudimit/MUSIC_dataset).

## Prerequisites

```sh
pip install youtube_dl tqdm
```

## Download the dataset

```sh
python download.py -i MUSIC_solo_videos.json -o data/MUSIC/solo
```
