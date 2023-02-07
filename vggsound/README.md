# Downloading the VGGSound Dataset

This folder contains the scripts for downloading the VGGSound dataset. The CSV file is downloaded from the original [repository](https://www.robots.ox.ac.uk/~vgg/data/vggsound/).

## Prerequisites

Shuffle and split the CSV file as follows.
Put `vggsound.csv` into your data directory, e.g. `data/vggsound`.

```sh
shuf data/vggsound/vggsound.csv > data/vggsound/vggsound-shuf.csv
split -l 10000 -d --additional-suffix=.csv data/vggsound/vggsound-shuf.csv data/vggsound/vggsound-shuf-
```
Install packeages
```sh
pip install youtube_dl tqdm pafy
```


## Download the dataset

Run the following script over all the CSV files.

```sh
python download_ffmpeg.py -e -s -i data/vggsound/vggsound-shuf-00.csv -o data/vggsound/vggsound/00/
```
