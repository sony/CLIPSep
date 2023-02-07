"""Datasets."""
import argparse
import csv
import logging
import pathlib
import random

import librosa
import numpy as np
import scipy.io.wavfile
import torch
import torchvision
from PIL import Image

import utils


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def transform():
    """Preprocessing transformations used in the CLIP model."""
    return torchvision.transforms.Compose(
        [
            _convert_image_to_rgb,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class MixDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filename,
        split,
        n_mix=2,
        audio_len=80000,
        audio_rate=16000,
        n_fft=1024,
        hop_len=256,
        win_len=1024,
        n_frames=3,
        stride_frames=1,
        img_size=224,
        fps=1,
        preprocess_func=None,
        max_sample=None,
        return_waveform=True,
        repeat=None,
        frame_margin=None,
        audio_only=False,
    ):
        assert split in (
            "train",
            "valid",
        ), "`split` must be one of 'train' or 'valid'."

        super().__init__()
        self.split = split
        self.n_mix = n_mix
        self.audio_len = audio_len
        self.audio_rate = audio_rate
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.n_frames = n_frames
        self.stride_frames = stride_frames
        self.img_size = img_size
        self.fps = fps
        self.preprocess_func = preprocess_func
        self.return_waveform = return_waveform
        self.frame_margin = frame_margin
        self.audio_only = audio_only

        # Compute useful numbers
        self.audio_sec = 1.0 * self.audio_len / self.audio_rate
        self.HS = self.n_fft // 2 + 1
        self.WS = (self.audio_len + 1) // self.hop_len

        # Read samples
        self.samples = []
        for row in csv.reader(open(filename, "r"), delimiter=","):
            # Skip bad rows
            if len(row) < 2:
                continue
            self.samples.append(row)

        # Check number of samples
        assert len(self.samples) > 0, "No samples found!"

        # Repeat the sample list if necessary
        if repeat is not None:
            self.samples *= repeat

        # Set max number of samples
        if max_sample is not None:
            self.samples = self.samples[:max_sample]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        N = self.n_mix
        frames = [None] * N
        audios = [None] * N
        infos = [[] for _ in range(N)]
        filenames_frame = [[] for _ in range(N)]
        filenames_audio = [""] * N
        center_frames = [0] * N

        # Get the first video
        infos[0] = self.samples[idx]

        # Sample other videos
        if self.split != "train":
            random.seed(idx)
        candidates = list(range(idx))
        candidates.extend(range(idx + 1, len(self.samples)))
        sampled = random.sample(candidates, N - 1)
        for n in range(1, N):
            infos[n] = self.samples[sampled[n - 1]]

        # select frames
        if self.frame_margin is None:
            idx_margin = int(self.fps * 3)
        else:
            idx_margin = self.frame_margin
        for n, (filename_audio, filename_frame, total_frames, _) in enumerate(
            infos
        ):
            if self.split == "train":
                # Randomly select a center frame, excluding the start and
                # ending n frames
                center_frameN = random.randint(
                    idx_margin + 1, int(total_frames) - idx_margin
                )
            else:
                center_frameN = int(total_frames) // 2
            center_frames[n] = center_frameN

            # absolute frame/audio paths
            if not self.audio_only:
                for i in range(self.n_frames):
                    idx_offset = (i - self.n_frames // 2) * self.stride_frames
                    filenames_frame[n].append(
                        f"{filename_frame}/{center_frameN + idx_offset:06d}.jpg"
                    )
            filenames_audio[n] = filename_audio

        # Load the data
        try:
            for n in range(N):
                if not self.audio_only:
                    frames[n] = self._load_frames(filenames_frame[n])
                center_time = (center_frames[n] - 0.5) / self.fps
                audios[n] = self._load_audio(filenames_audio[n], center_time)

            # Divide the waveforms by N
            for n in range(N):
                audios[n] /= N

            if self.return_waveform:
                # Compute audio mixture
                audio_mix = sum(audios)
            else:
                # Compute STFT
                spec_mix = 0
                mags = []
                for n in range(N):
                    spec = librosa.stft(
                        audios[n],
                        n_fft=self.n_fft,
                        hop_length=self.hop_len,
                        win_length=self.win_len,
                    )
                    spec_mix += spec
                    mags.append(torch.tensor(np.abs(spec)).unsqueeze(0))

                # Compute magnitude and phase mixture
                mag_mix = torch.tensor(np.abs(spec_mix)).unsqueeze(0)
                phase_mix = torch.tensor(np.angle(spec_mix)).unsqueeze(0)

            # Convert into torch tensors
            for n in range(N):
                audios[n] = torch.tensor(audios[n])
            if self.return_waveform:
                audio_mix = torch.tensor(audio_mix)

        except Exception as e:
            logging.debug(f"Failed loading frame/audio: {e}")
            # Create dummy data if failed
            frames = [
                torch.zeros(self.n_frames, 3, self.img_size, self.img_size)
                for _ in range(N)
            ]
            audios = [torch.zeros(self.audio_len) for _ in range(N)]
            mags = [torch.zeros(1, self.HS, self.WS) for _ in range(N)]
            if self.return_waveform:
                audio_mix = torch.zeros(self.audio_len)
            else:
                mag_mix = torch.zeros(1, self.HS, self.WS)
                phase_mix = torch.zeros(1, self.HS, self.WS)

        ret_dict = {"infos": infos}
        if not self.audio_only:
            ret_dict["frames"] = frames
        if self.return_waveform:
            ret_dict["audio_mix"] = audio_mix
            ret_dict["audios"] = audios
        else:
            ret_dict["mag_mix"] = mag_mix
            ret_dict["mags"] = mags
        if self.split != "train":
            ret_dict["audios"] = audios
            if not self.return_waveform:
                ret_dict["phase_mix"] = phase_mix

        return ret_dict

    def _load_frames(self, filenames):
        frames = [
            Image.open(filename).convert("RGB") for filename in filenames
        ]
        if self.preprocess_func is None:
            return torch.stack(frames)
        return torch.stack([self.preprocess_func(frame) for frame in frames])

    def _load_audio(self, filename, center_time):
        # Initialize an empty audio array
        audio = np.zeros(self.audio_len, dtype=np.float32)

        # Load the audio
        rate, audio_raw = scipy.io.wavfile.read(filename)
        audio_raw = torch.tensor(audio_raw / -np.iinfo(np.int16).min)

        # Check sampling rate
        assert rate == self.audio_rate, (
            f"Found an unexpected sampling rate of {rate} for {filename} "
            f"(expected {self.audio_rate})"
        )

        # Repeat if audio is too short
        if audio_raw.shape[0] < rate * self.audio_sec:
            repeats = int(rate * self.audio_sec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, repeats)

        # Crop N seconds
        len_raw = audio_raw.shape[0]
        center = int(center_time * self.audio_rate)
        start = max(0, center - self.audio_len // 2)
        end = min(len_raw, center + self.audio_len // 2)

        audio[
            (self.audio_len // 2 - (center - start)) : (
                self.audio_len // 2 + (end - center)
            )
        ] = audio_raw[start:end]

        # Randomize the volume for training
        if self.split == "train":
            audio *= random.random() + 0.5  # 0.5-1.5

        # Clip the audio to [-1, 1]
        audio = np.clip(audio, -1, 1)

        return audio


class MixDatasetV2(torch.utils.data.Dataset):
    def __init__(
        self,
        filename,
        filename2,
        split,
        audio_len=80000,
        audio_rate=16000,
        n_fft=1024,
        hop_len=256,
        win_len=1024,
        n_frames=3,
        stride_frames=1,
        img_size=224,
        fps=1,
        preprocess_func=None,
        max_sample=None,
        return_waveform=True,
        repeat=None,
        frame_margin=None,
        audio_only=False,
        N_test_sources=2,
        normalize=False
    ):
        assert split in (
            "train",
            "valid",
        ), "`split` must be one of 'train' or 'valid'."

        super().__init__()
        self.split = split
        self.audio_len = audio_len
        self.audio_rate = audio_rate
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.n_frames = n_frames
        self.stride_frames = stride_frames
        self.img_size = img_size
        self.fps = fps
        self.preprocess_func = preprocess_func
        self.return_waveform = return_waveform
        self.frame_margin = frame_margin
        self.audio_only = audio_only
        self.N_test_sources = N_test_sources
        self.normalize = normalize

        # Compute useful numbers
        self.audio_sec = 1.0 * self.audio_len / self.audio_rate
        self.HS = self.n_fft // 2 + 1
        self.WS = (self.audio_len + 1) // self.hop_len

        # Read samples
        self.samples = []
        for row in csv.reader(open(filename, "r"), delimiter=","):
            # Skip bad rows
            if len(row) < 2:
                continue
            self.samples.append(row)

        # Read samples
        self.samples2 = []
        for row in csv.reader(open(filename2, "r"), delimiter=","):
            # Skip bad rows
            if len(row) < 2:
                continue
            self.samples2.append(row)

        # Check number of samples
        assert len(self.samples) > 0, f"No samples found for {filename}."
        assert len(self.samples2) > 0, f"No samples found for {filename2}."

        # Repeat the sample list if necessary
        if repeat is not None:
            self.samples *= repeat

        # Set max number of samples
        if max_sample is not None:
            self.samples = self.samples[:max_sample]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        N = self.N_test_sources
        frames = None
        audios = [None] * N
        infos = [[] for _ in range(N)]
        filenames_frame = [[] for _ in range(N)]
        filenames_audio = [""] * N
        center_frames = [0] * N

        # Get the first video
        infos[0] = self.samples[idx]

        # Sample other videos
        if self.split != "train":
            random.seed(idx)
        for i in range(1,N):
            infos[i] = self.samples2[random.randint(0, len(self.samples2) - 1)]

        # select frames
        if self.frame_margin is None:
            idx_margin = int(self.fps * 3)
        else:
            idx_margin = self.frame_margin
        for n, (filename_audio, filename_frame, total_frames, _) in enumerate(
            infos
        ):
            if self.split == "train":
                # Randomly select a center frame, excluding the start and
                # ending n frames
                center_frameN = random.randint(
                    idx_margin + 1, int(total_frames) - idx_margin
                )
            else:
                center_frameN = int(total_frames) // 2
            center_frames[n] = center_frameN

            # absolute frame/audio paths
            if not self.audio_only:
                for i in range(self.n_frames):
                    idx_offset = (i - self.n_frames // 2) * self.stride_frames
                    filenames_frame[n].append(
                        f"{filename_frame}/{center_frameN + idx_offset:06d}.jpg"
                    )
            filenames_audio[n] = filename_audio

        # Load the data
        try:
            if not self.audio_only:
                frames = [self._load_frames(filenames_frame[0])]
            for n in range(N):
                center_time = (center_frames[n] - 0.5) / self.fps
                audios[n] = self._load_audio(filenames_audio[n], center_time)

            if N>2:
                audios[1] = np.mean(audios[1:], axis=0)
                audios = audios[:2]
                N=2

            # normalize audio to have target SNR
            if self.normalize:
                audios = self.normalize_to_target_snr(audios, 3)

            # Divide the waveforms by N
            for n in range(N):
                audios[n] /= N

            if self.return_waveform:
                # Compute audio mixture
                audio_mix = sum(audios)
            else:
                # Compute STFT
                spec_mix = 0
                mags = []
                for n in range(N):
                    spec = librosa.stft(
                        audios[n],
                        n_fft=self.n_fft,
                        hop_length=self.hop_len,
                        win_length=self.win_len,
                    )
                    spec_mix += spec
                    mags.append(torch.tensor(np.abs(spec)).unsqueeze(0))

                # Compute magnitude and phase mixture
                mag_mix = torch.tensor(np.abs(spec_mix)).unsqueeze(0)
                phase_mix = torch.tensor(np.angle(spec_mix)).unsqueeze(0)

            # Convert into torch tensors
            for n in range(N):
                audios[n] = torch.tensor(audios[n])
            if self.return_waveform:
                audio_mix = torch.tensor(audio_mix)

        except Exception as e:
            logging.debug(f"Failed loading frame/audio: {e}")
            # Create dummy data if failed
            if not self.audio_only:
                frames = [
                    torch.zeros(self.n_frames, 3, self.img_size, self.img_size)
                ]
            audios = [torch.zeros(self.audio_len) for _ in range(N)]
            mags = [torch.zeros(1, self.HS, self.WS) for _ in range(N)]
            if self.return_waveform:
                audio_mix = torch.zeros(self.audio_len)
            else:
                mag_mix = torch.zeros(1, self.HS, self.WS)
                phase_mix = torch.zeros(1, self.HS, self.WS)

        ret_dict = {"infos": infos}
        if not self.audio_only:
            ret_dict["frames"] = frames
        if self.return_waveform:
            ret_dict["audio_mix"] = audio_mix
            ret_dict["audios"] = audios
        else:
            ret_dict["mag_mix"] = mag_mix
            ret_dict["mags"] = mags
        if self.split != "train":
            ret_dict["audios"] = audios
            if not self.return_waveform:
                ret_dict["phase_mix"] = phase_mix

        return ret_dict

    def _load_frames(self, filenames):
        frames = [
            Image.open(filename).convert("RGB") for filename in filenames
        ]
        if self.preprocess_func is None:
            return torch.stack(frames)
        return torch.stack([self.preprocess_func(frame) for frame in frames])

    def _load_audio(self, filename, center_time):
        # Initialize an empty audio array
        audio = np.zeros(self.audio_len, dtype=np.float32)

        # Load the audio
        rate, audio_raw = scipy.io.wavfile.read(filename)
        audio_raw = torch.tensor(audio_raw / -np.iinfo(np.int16).min)

        # Check sampling rate
        assert rate == self.audio_rate, (
            f"Found an unexpected sampling rate of {rate} for {filename} "
            f"(expected {self.audio_rate})"
        )

        # Repeat if audio is too short
        if audio_raw.shape[0] < rate * self.audio_sec:
            repeats = int(rate * self.audio_sec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, repeats)

        # Crop N seconds
        len_raw = audio_raw.shape[0]
        center = int(center_time * self.audio_rate)
        start = max(0, center - self.audio_len // 2)
        end = min(len_raw, center + self.audio_len // 2)

        audio[
            (self.audio_len // 2 - (center - start)) : (
                self.audio_len // 2 + (end - center)
            )
        ] = audio_raw[start:end]

        # Randomize the volume for training
        if self.split == "train":
            audio *= random.random() + 0.5  # 0.5-1.5

        # Clip the audio to [-1, 1]
        audio = np.clip(audio, -1, 1)

        return audio

    def normalize_to_target_snr(self, audios, target_snr=0):
        eps = 1e-8
        audios[0] = audios[0]/np.max(np.abs(audios[0])+eps)
        es = [np.sum(audio**2) for audio in audios]
        e_s = es[0]
        es_n = es[1:]

        for i, e_n in enumerate(es_n):
            snr = 10*np.log10(e_s/(e_n+eps))
            w = np.power(10, snr/10/target_snr)
            w = np.clip(w, 0.25, 4)
            audios[i+1] = audios[i+1]*w

        return audios


class SingleDatasetV2(torch.utils.data.Dataset):
    def __init__(
        self,
        filename,
        split,
        audio_len=80000,
        audio_rate=16000,
        n_fft=1024,
        hop_len=256,
        win_len=1024,
        n_frames=3,
        stride_frames=1,
        img_size=224,
        fps=1,
        preprocess_func=None,
        max_sample=None,
        return_waveform=True,
        repeat=None,
        frame_margin=None,
        audio_only=False,
        N_test_sources=2,

    ):
        assert split in (
            "train",
            "valid",
        ), "`split` must be one of 'train' or 'valid'."

        super().__init__()
        self.split = split
        self.audio_len = audio_len
        self.audio_rate = audio_rate
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.n_frames = n_frames
        self.stride_frames = stride_frames
        self.img_size = img_size
        self.fps = fps
        self.preprocess_func = preprocess_func
        self.return_waveform = return_waveform
        self.frame_margin = frame_margin
        self.audio_only = audio_only
        self.N_test_sources = N_test_sources

        # Compute useful numbers
        self.audio_sec = 1.0 * self.audio_len / self.audio_rate
        self.HS = self.n_fft // 2 + 1
        self.WS = (self.audio_len + 1) // self.hop_len

        # Read samples
        self.samples = []
        for row in csv.reader(open(filename, "r"), delimiter=","):
            # Skip bad rows
            if len(row) < 2:
                continue
            self.samples.append(row)

        # Check number of samples
        assert len(self.samples) > 0, f"No samples found for {filename}."

        # Repeat the sample list if necessary
        if repeat is not None:
            self.samples *= repeat

        # Set max number of samples
        if max_sample is not None:
            self.samples = self.samples[:max_sample]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        N = 1
        frames = None
        audios = [None] * N
        infos = [[] for _ in range(N)]
        filenames_frame = [[] for _ in range(N)]
        filenames_audio = [""] * N
        center_frames = [0] * N

        # Get the first video
        infos[0] = self.samples[idx]

        # Sample other videos
        if self.split != "train":
            random.seed(idx)

        # select frames
        if self.frame_margin is None:
            idx_margin = int(self.fps * 3)
        else:
            idx_margin = self.frame_margin
        for n, (filename_audio, filename_frame, total_frames, _) in enumerate(
            infos
        ):
            if self.split == "train":
                # Randomly select a center frame, excluding the start and
                # ending n frames
                center_frameN = random.randint(
                    idx_margin + 1, int(total_frames) - idx_margin
                )
            else:
                center_frameN = int(total_frames) // 2
            center_frames[n] = center_frameN

            # absolute frame/audio paths
            if not self.audio_only:
                for i in range(self.n_frames):
                    idx_offset = (i - self.n_frames // 2) * self.stride_frames
                    filenames_frame[n].append(
                        f"{filename_frame}/{center_frameN + idx_offset:06d}.jpg"
                    )
            filenames_audio[n] = filename_audio

        # Load the data
        try:
            if not self.audio_only:
                frames = [self._load_frames(filenames_frame[0])]
            for n in range(N):
                center_time = (center_frames[n] - 0.5) / self.fps
                audios[n] = self._load_audio(filenames_audio[n], center_time)

            if not self.return_waveform:
                # Compute STFT
                spec_mix = 0
                mags = []
                for n in range(N):
                    spec = librosa.stft(
                        audios[n],
                        n_fft=self.n_fft,
                        hop_length=self.hop_len,
                        win_length=self.win_len,
                    )
                    spec_mix += spec
                    mags.append(torch.tensor(np.abs(spec)).unsqueeze(0))

                # Compute magnitude and phase mixture
                mag_mix = torch.tensor(np.abs(spec_mix)).unsqueeze(0)
                phase_mix = torch.tensor(np.angle(spec_mix)).unsqueeze(0)

            # Convert into torch tensors
            for n in range(N):
                audios[n] = torch.tensor(audios[n])
            if self.return_waveform:
                audio_mix = torch.tensor(audio_mix)

        except Exception as e:
            logging.debug(f"Failed loading frame/audio: {e}")
            # Create dummy data if failed
            if not self.audio_only:
                frames = [
                    torch.zeros(self.n_frames, 3, self.img_size, self.img_size)
                ]
            audios = [torch.zeros(self.audio_len) for _ in range(N)]
            mags = [torch.zeros(1, self.HS, self.WS) for _ in range(N)]
            if self.return_waveform:
                audio_mix = torch.zeros(self.audio_len)
            else:
                mag_mix = torch.zeros(1, self.HS, self.WS)
                phase_mix = torch.zeros(1, self.HS, self.WS)

        ret_dict = {"infos": infos}
        if not self.audio_only:
            ret_dict["frames"] = frames
        if self.return_waveform:
            ret_dict["audio_mix"] = audio_mix
            ret_dict["audios"] = audios
        else:
            ret_dict["mag_mix"] = mag_mix
            ret_dict["mags"] = mags
        if self.split != "train":
            ret_dict["audios"] = audios
            if not self.return_waveform:
                ret_dict["phase_mix"] = phase_mix

        return ret_dict

    def _load_frames(self, filenames):
        frames = [
            Image.open(filename).convert("RGB") for filename in filenames
        ]
        if self.preprocess_func is None:
            return torch.stack(frames)
        return torch.stack([self.preprocess_func(frame) for frame in frames])

    def _load_audio(self, filename, center_time):
        # Initialize an empty audio array
        audio = np.zeros(self.audio_len, dtype=np.float32)

        # Load the audio
        rate, audio_raw = scipy.io.wavfile.read(filename)
        audio_raw = torch.tensor(audio_raw / -np.iinfo(np.int16).min)

        # Check sampling rate
        assert rate == self.audio_rate, (
            f"Found an unexpected sampling rate of {rate} for {filename} "
            f"(expected {self.audio_rate})"
        )

        # Repeat if audio is too short
        if audio_raw.shape[0] < rate * self.audio_sec:
            repeats = int(rate * self.audio_sec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, repeats)

        # Crop N seconds
        len_raw = audio_raw.shape[0]
        center = int(center_time * self.audio_rate)
        start = max(0, center - self.audio_len // 2)
        end = min(len_raw, center + self.audio_len // 2)

        audio[
            (self.audio_len // 2 - (center - start)) : (
                self.audio_len // 2 + (end - center)
            )
        ] = audio_raw[start:end]

        # Randomize the volume for training
        if self.split == "train":
            audio *= random.random() + 0.5  # 0.5-1.5

        # Clip the audio to [-1, 1]
        audio = np.clip(audio, -1, 1)

        return audio


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--list_train",
        type=pathlib.Path,
        help="filename of the training list",
    )
    parser.add_argument(
        "-v",
        "--list_val",
        type=pathlib.Path,
        help="filename of the validation list",
    )
    parser.add_argument(
        "--n_valid", type=int, help="number of images to evaluate"
    )

    # Data
    parser.add_argument(
        "--n_mix", default=2, type=int, help="number of sounds to mix"
    )
    parser.add_argument(
        "--frames", default=3, type=int, help="number of frames"
    )
    parser.add_argument(
        "--stride_frames",
        default=1,
        type=int,
        help="sampling stride of frames",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="batch size per GPU",
    )
    parser.add_argument(
        "--frame_margin",
        type=int,
        help="the number of starting and ending frames to exclude",
    )
    parser.add_argument(
        "--audio_only",
        action="store_true",
        help="whether the dataset contains only audio",
    )

    # Audio configurations
    parser.add_argument(
        "--audio_len", default=80000, type=int, help="audio length (samples)"
    )
    parser.add_argument(
        "--audio_rate", default=16000, type=int, help="sampling rate"
    )
    parser.add_argument(
        "--n_fft", default=1024, type=int, help="n_fft for STFT"
    )
    parser.add_argument(
        "--hop_len", default=256, type=int, help="hop length for STFT"
    )
    parser.add_argument(
        "--win_len", default=1024, type=int, help="window length for STFT"
    )

    # Image configurations
    parser.add_argument(
        "--img_size", default=224, type=int, help="size of input frame"
    )
    parser.add_argument(
        "--fps",
        default=1,
        type=float,
        help="video frame sampling rate",
    )

    # Others
    parser.add_argument(
        "--workers", default=8, type=int, help="number of data loading workers"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


def main():
    """Main function."""
    import pprint
    import sys
    import time

    import torch.utils.data
    import tqdm

    # Parse command-lind arguments
    args = parse_args()

    # Set up a console logger
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Create the training dataset
    logging.info("Creating the training dataset...")
    train_dataset = MixDataset(
        args.list_train,
        "train",
        n_mix=args.n_mix,
        audio_len=args.audio_len,
        audio_rate=args.audio_rate,
        n_fft=args.n_fft,
        hop_len=args.hop_len,
        win_len=args.win_len,
        n_frames=args.frames,
        stride_frames=args.stride_frames,
        img_size=args.img_size,
        fps=args.fps,
        preprocess_func=transform(),
        max_sample=None,
        return_waveform=False,
        frame_margin=args.frame_margin,
        audio_only=args.audio_only,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    n_batches_train = len(train_dataset) // args.batch_size
    logging.info(f"Number of batches: {n_batches_train}")

    # Show example batch
    logging.info("Example batch:")
    batch = next(iter(train_dataset))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            logging.info(f"Size of {k}: {v.size()}")
        elif isinstance(v, list):
            if isinstance(v[0], list):
                logging.info(f"{k}: {v}")
            else:
                logging.info(
                    f"Size of {k}: [{', '.join(str(x.size()) for x in v)}]"
                )
        else:
            logging.info(f"{k}: {len(v)}")

    # Iterate over the training set loader
    start = time.time()
    for _ in tqdm.tqdm(train_loader, ncols=120):
        pass
    time_elapsed = time.time() - start
    logging.info(f"Time elapsed: {time_elapsed}")
    logging.info(f"Time per batch: {time_elapsed / n_batches_train}")

    # Create the validation dataset
    logging.info("Creating the validation dataset...")
    val_dataset = MixDataset(
        args.list_val,
        split="valid",
        n_mix=args.n_mix,
        audio_len=args.audio_len,
        audio_rate=args.audio_rate,
        n_fft=args.n_fft,
        hop_len=args.hop_len,
        win_len=args.win_len,
        n_frames=args.frames,
        stride_frames=args.stride_frames,
        img_size=args.img_size,
        fps=args.fps,
        preprocess_func=transform(),
        max_sample=None,
        return_waveform=False,
        frame_margin=args.frame_margin,
        audio_only=args.audio_only,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )
    n_batches_val = len(val_dataset) // args.batch_size
    logging.info(f"Number of batches: {n_batches_val}")

    # Show example batch
    logging.info("Example batch:")
    batch = next(iter(val_dataset))
    for k, v in batch.items():
        if k == "infos":
            continue
        if isinstance(v, torch.Tensor):
            logging.info(f"Size of {k}: {v.size()}")
        elif isinstance(v, list):
            logging.info(
                f"Size of {k}: [{', '.join(str(x.size()) for x in v)}]"
            )
        else:
            logging.info(f"{k}: {len(v)}")

    # Iterate over the validation set loader
    start = time.time()
    for _ in tqdm.tqdm(val_loader, ncols=120):
        pass
    time_elapsed = time.time() - start
    logging.info(f"Time elapsed: {time_elapsed}")
    logging.info(f"Time per batch: {time_elapsed / n_batches_val}")


if __name__ == "__main__":
    main()
