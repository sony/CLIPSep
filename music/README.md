# Downloading the MUSIC Dataset

This folder contains the scripts for downloading the MUSIC dataset. The JSON files are downloaded from the original [repository](https://github.com/roudimit/MUSIC_dataset).

## Prerequisites

```sh
pip install youtube_dl tqdm
```

## Download the dataset

```sh
python download.py -i MUSIC_solo_videos.json -o data/MUSIC/solo/video
```

## preprocess data

Extract audio from videos
```sh
python extract_audio.py -i data/MUSIC/solo/video -o data/MUSIC/solo/audio -s -e
```
Extract image frames from videos
```sh
python extract_frames.py -i data/MUSIC/solo/video -o data/MUSIC/solo/frames -s -e
```
Resize and crop images
```sh
python preprocess.py -i data/MUSIC/solo/frames -o data/MUSIC/solo/preprocessed -s -e
```