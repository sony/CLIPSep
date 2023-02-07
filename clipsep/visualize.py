"""Visualize the results."""
import argparse
import logging
import pathlib
import pprint
import random
import sys
import types

import clip
import imageio
import numpy as np
import scipy.io.wavfile
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision
import tqdm

import clipsep
import dataset
import utils
import visualization


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "-v", "--vis_dir", type=pathlib.Path, help="visualization directory"
    )
    parser.add_argument(
        "-t",
        "--test_list",
        type=pathlib.Path,
        help="filename of the test list",
    )
    parser.add_argument(
        "-t2",
        "--test_list2",
        type=pathlib.Path,
        help="filename of the test list",
    )
    parser.add_argument(
        "-n_vis",
        "--n_visualization",
        default=64,
        type=int,
        help="number of samples to visualize",
    )
    parser.add_argument(
        "--model_steps",
        type=int,
        help="step of the trained model to load (default to the best model)",
    )
    parser.add_argument(
        "-l", "--log_filename", type=pathlib.Path, help="log filename"
    )
    parser.add_argument(
        "--binary",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="whether to binarize the masks",
    )
    parser.add_argument(
        "--threshold", default=0.5, type=float, help="binarization threshold"
    )
    parser.add_argument(
        "--prompt_ens",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="whether to ensemble prompts",
    )

    # Data
    parser.add_argument(
        "--batch_size", default=32, type=int, help="batch size"
    )
    parser.add_argument(
        "--audio_only",
        action="store_true",
        help="whether the dataset contains only audio",
    )

    # Others
    parser.add_argument("--seed", default=1234, type=int, help="manual seed")
    parser.add_argument(
        "--gpus", default=1, type=int, help="number of gpus to use"
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        help="number of data loading workers",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )

    return parser.parse_args(args=args, namespace=namespace)


def get_text_prompt(label):
    """Get the text prompt for a label."""
    return f"a photo of {label}"

def get_text_prompts(label):
    """Get the text prompt for a label."""
    return [f"a photo of {label}.", f"a photo of the small {label}.", f"a low resolution photo of a {label}.", f"a photo of many {label}."]


def count_parameters(net):
    """Return the number of parameters in a model."""
    return sum(p.numel() for p in net.parameters())


def output_visuals(
    html_writer,
    batch,
    out,
    sample_dir,
    mode,
    n_mix=2,
    audio_rate=16000,
    n_fft=1024,
    hop_len=256,
    win_len=1024,
    n_frames=3,
    stride_frames=1,
    fps=1,
    use_log_freq=True,
    use_binary_mask=True,
    image_model="clip",
):
    """Visualize the results."""
    # Fetch data and predictions
    mag_mix = batch["mag_mix"]
    phase_mix = batch["phase_mix"]
    frames = batch.get("frames")
    if frames is not None:
        frames = frames[0]
    infos = batch["infos"]
    pred_masks_ = out["pred_masks"][0]
    gt_masks_ = out["gt_masks"][0]
    int_masks_ = out["gt_masks"][1]
    pit_masks_ = out.get("pit_masks")
    mag_mix_ = out["mag_mix"]
    weight_ = out["weight"]

    if "clippit" in image_model:
        pred_masks_ = torch.sigmoid(pred_masks_)
        pit_masks_ = [torch.sigmoid(mask) for mask in pit_masks_]

    # Unwarp log scale
    N = n_mix
    B = mag_mix.size(0)

    if use_log_freq:
        grid_unwarp = torch.from_numpy(
            utils.warpgrid(
                B,
                n_fft // 2 + 1,
                gt_masks_.size(3),
                warp=False,
            )
        ).to(pred_masks_.device)
        pred_masks_linear = F.grid_sample(
            pred_masks_, grid_unwarp, align_corners=True
        )
        gt_masks_linear = F.grid_sample(
            gt_masks_, grid_unwarp, align_corners=True
        )
        int_masks_linear = F.grid_sample(
            int_masks_, grid_unwarp, align_corners=True
        )
    else:
        pred_masks_linear = pred_masks_
        gt_masks_linear = gt_masks_
        int_masks_linear = int_masks_

    pit_masks_linear = [None] * N
    for n in range(N):
        if use_log_freq:
            grid_unwarp = torch.from_numpy(
                utils.warpgrid(
                    B,
                    n_fft // 2 + 1,
                    gt_masks_.size(3),
                    warp=False,
                )
            ).to(pred_masks_[n].device)
            if pit_masks_ is not None:
                pit_masks_linear[n] = F.grid_sample(
                    pit_masks_[n], grid_unwarp, align_corners=True
                )
        else:
            if pit_masks_ is not None:
                pit_masks_linear[n] = pit_masks_[n]

    # Convert into numpy arrays
    mag_mix = mag_mix.detach().cpu().numpy()
    mag_mix_ = mag_mix_.detach().cpu().numpy()
    phase_mix = phase_mix.detach().cpu().numpy()
    weight_ = weight_.detach().cpu().numpy()

    pred_masks_ = pred_masks_.detach().cpu().numpy()
    pred_masks_linear = pred_masks_linear.detach().cpu().numpy()
    gt_masks_ = gt_masks_.detach().cpu().numpy()
    gt_masks_linear = gt_masks_linear.detach().cpu().numpy()
    int_masks_ = int_masks_.detach().cpu().numpy()
    int_masks_linear = int_masks_linear.detach().cpu().numpy()
    # Apply the threshold
    if use_binary_mask:
        pred_masks_[n] = (pred_masks_[n] > 0.5).astype(np.float32)
        pred_masks_linear[n] = (pred_masks_linear[n] > 0.5).astype(np.float32)

    for n in range(N):
        if pit_masks_ is not None:
            pit_masks_[n] = pit_masks_[n].detach().cpu().numpy()
            pit_masks_linear[n] = pit_masks_linear[n].detach().cpu().numpy()
            # Apply the threshold
            if use_binary_mask:
                pit_masks_linear[n] = (pit_masks_linear[n] > 0.5).astype(
                    np.float32
                )

    # Loop over each sample
    for j in range(B):
        # Initialize row elements
        elems = []

        # Video names
        prefix = []
        for n in range(N):
            prefix.append(
                "-".join(infos[n][0][j].split("/")[-2:]).split(".")[0]
            )
        prefix = "+".join(prefix)
        (sample_dir / prefix).mkdir(exist_ok=True)

        # Save the mixture
        mix_wav = utils.istft_reconstruction(
            mag_mix[j, 0], phase_mix[j, 0], hop_len=hop_len, win_len=win_len
        )
        mix_amp = utils.magnitude2heatmap(mag_mix[j, 0])
        weight = utils.magnitude2heatmap(weight_[j, 0], log=False, scale=100.0)
        filename_mixwav = f"{prefix}/mix.wav"
        filename_mixmag = f"{prefix}/mix.png"
        filename_weight = f"{prefix}/weight.png"
        imageio.imwrite(sample_dir / filename_mixmag, mix_amp[::-1, :])
        imageio.imwrite(sample_dir / filename_weight, weight[::-1, :])
        scipy.io.wavfile.write(
            sample_dir / filename_mixwav, audio_rate, mix_wav
        )
        elems.extend(
            [
                {"text": prefix},
                {"image": filename_mixmag, "audio": filename_mixwav},
            ]
        )

        # Save each component

        # GT and predicted audio recovery
        gt_mag = mag_mix[j, 0] * gt_masks_linear[j, 0]
        gt_wav = utils.istft_reconstruction(
            gt_mag,
            phase_mix[j, 0],
            hop_len=hop_len,
            win_len=win_len,
        )
        pred_mag = mag_mix[j, 0] * pred_masks_linear[j, 0]
        preds_wav = utils.istft_reconstruction(
            pred_mag,
            phase_mix[j, 0],
            hop_len=hop_len,
            win_len=win_len,
        )

        # Save masks
        filename_gtmask = f"{prefix}/gtmask.png"
        filename_predmask = f"{prefix}/predmask.png"
        gt_mask = (np.clip(gt_masks_[j, 0], 0, 1) * 255).astype(np.uint8)
        pred_mask = (np.clip(pred_masks_[j, 0], 0, 1) * 255).astype(np.uint8)
        imageio.imwrite(sample_dir / filename_gtmask, gt_mask[::-1, :])
        imageio.imwrite(sample_dir / filename_predmask, pred_mask[::-1, :])

        # Save spectrogram (log of magnitude, show colormap)
        filename_gtmag = f"{prefix}/gtamp.png"
        filename_predmag = f"{prefix}/predamp.png"
        gt_mag = utils.magnitude2heatmap(gt_mag)
        pred_mag = utils.magnitude2heatmap(pred_mag)
        imageio.imwrite(sample_dir / filename_gtmag, gt_mag[::-1, :, :])
        imageio.imwrite(sample_dir / filename_predmag, pred_mag[::-1, :, :])

        # Save audios
        filename_gtwav = f"{prefix}/gt.wav"
        filename_predwav = f"{prefix}/pred.wav"
        scipy.io.wavfile.write(sample_dir / filename_gtwav, audio_rate, gt_wav)
        scipy.io.wavfile.write(
            sample_dir / filename_predwav, audio_rate, preds_wav
        )

        if frames is not None:
            # Save videos
            frames_tensor = [
                utils.recover_rgb_clip(frames[j, t]) for t in range(n_frames)
            ]
            frames_tensor = np.asarray(frames_tensor)
            filename_video = str(sample_dir / prefix / f"video.mp4")
            utils.save_video(
                filename_video, frames_tensor, fps=fps / stride_frames
            )

            # Save frames
            (sample_dir / prefix / f"frames").mkdir(exist_ok=True)
            for i, frame_tensor in enumerate(frames_tensor):
                imageio.imwrite(
                    sample_dir / f"{prefix}/frames/frames_{i}.png",
                    frame_tensor,
                )

            # Combine ground truth video and audio
            filename_av = f"{prefix}/av.mp4"
            utils.combine_video_audio(
                filename_video,
                sample_dir / filename_gtwav,
                sample_dir / filename_av,
            )

        if mode == "image":
            elems.append({"video": filename_av})
        elif mode == "text":
            # prompt = get_text_prompt(batch["infos"][n][3][j])
            prompt = get_text_prompt(batch["infos"][0][3][j])
            elems.append({"text": prompt})
        elems.extend(
            [
                {"image": filename_predmag, "audio": filename_predwav},
                {"image": filename_gtmag, "audio": filename_gtwav},
                {"image": filename_predmask},
                {"image": filename_gtmask},
            ]
        )

        n = 1

        # GT and predicted audio recovery
        int_mag = mag_mix[j, 0] * int_masks_linear[j, 0]
        int_wav = utils.istft_reconstruction(
            int_mag,
            phase_mix[j, 0],
            hop_len=hop_len,
            win_len=win_len,
        )

        # Save masks
        filename_intmask = f"{prefix}/intmask.png"
        int_mask = (np.clip(int_masks_[j, 0], 0, 1) * 255).astype(np.uint8)
        imageio.imwrite(sample_dir / filename_intmask, int_mask[::-1, :])

        # Save spectrogram (log of magnitude, show colormap)
        filename_intmag = f"{prefix}/intamp.png"
        int_mag = utils.magnitude2heatmap(int_mag)
        imageio.imwrite(sample_dir / filename_intmag, int_mag[::-1, :, :])

        # Save audios
        filename_intwav = f"{prefix}/int.wav"
        scipy.io.wavfile.write(
            sample_dir / filename_intwav, audio_rate, int_wav
        )

        elems.extend(
            [
                {"image": filename_intmag, "audio": filename_intwav},
                {"image": filename_intmask},
            ]
        )

        elems.append({"image": filename_weight})

        if pit_masks_ is not None:

            # PIT audio recovery
            pit_mag0 = mag_mix[j, 0] * pit_masks_linear[0][j, 0]
            pit_mag1 = mag_mix[j, 0] * pit_masks_linear[1][j, 0]
            pit_wav0 = utils.istft_reconstruction(
                pit_mag0,
                phase_mix[j, 0],
                hop_len=hop_len,
                win_len=win_len,
            )
            pit_wav1 = utils.istft_reconstruction(
                pit_mag1,
                phase_mix[j, 0],
                hop_len=hop_len,
                win_len=win_len,
            )

            # Save masks
            filename_pitmask0 = f"{prefix}/pitmask1.png"
            filename_pitmask1 = f"{prefix}/pitmask2.png"
            pit_mask0 = (np.clip(pit_masks_[0][j, 0], 0, 1) * 255).astype(
                np.uint8
            )
            pit_mask1 = (np.clip(pit_masks_[1][j, 0], 0, 1) * 255).astype(
                np.uint8
            )
            imageio.imwrite(sample_dir / filename_pitmask0, pit_mask0[::-1, :])
            imageio.imwrite(sample_dir / filename_pitmask1, pit_mask1[::-1, :])

            # Save spectrogram (log of magnitude, show colormap)
            filename_pitmag0 = f"{prefix}/pitmag1.png"
            filename_pitmag1 = f"{prefix}/pitmag2.png"
            pit_mag0 = utils.magnitude2heatmap(pit_mag0)
            pit_mag1 = utils.magnitude2heatmap(pit_mag1)
            imageio.imwrite(
                sample_dir / filename_pitmag0, pit_mag0[::-1, :, :]
            )
            imageio.imwrite(
                sample_dir / filename_pitmag1, pit_mag1[::-1, :, :]
            )

            # Save audios
            filename_pitwav0 = f"{prefix}/pit0.wav"
            filename_pitwav1 = f"{prefix}/pit1.wav"
            scipy.io.wavfile.write(
                sample_dir / filename_pitwav0, audio_rate, pit_wav0
            )
            scipy.io.wavfile.write(
                sample_dir / filename_pitwav1, audio_rate, pit_wav1
            )

            elems.extend(
                [
                    {"image": filename_pitmag0, "audio": filename_pitwav0},
                    {"image": filename_pitmask0},
                    {"image": filename_pitmag1, "audio": filename_pitwav1},
                    {"image": filename_pitmask1},
                ]
            )

        html_writer.write_row(elems)


def new_clip_forward(self, image=None, text=None):
    """A CLIP forward function that automatically chooses the mode."""
    if image is None and text is None:
        raise ValueError("Either `image` or `text` must be given.")
    if image is None:
        return self.encode_text(text)
    if text is None:
        return self.encode_image(image)
    return self.old_forward(image, text)


def get_html_headers(n_mix, image_model):
    """Return the HTML headers."""
    headers = ["Filename", "Input Mixed Audio"]
    headers.extend(
        [
            f"Query",
            f"Predicted Audio",
            f"GroundTruth Audio",
            f"Predicted Mask",
            f"GroundTruth Mask",
            f"Interference Audio",
            f"Interference Mask",
        ]
    )
    headers.append("Loss weighting")
    if "clippit" in image_model:
        headers.extend(
            [
                "Predicted Audio 1 (PIT)",
                "Predicted Mask 1 (PIT)",
                "Predicted Audio 2 (PIT)",
                "Predicted Mask 2 (PIT)",
            ]
        )
    return headers


def main(args):
    """Main function."""
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Make sure the checkpoint and sample directories exist
    (args.vis_dir).mkdir(exist_ok=True)
    (args.vis_dir / "text").mkdir(exist_ok=True)
    (args.vis_dir / "image").mkdir(exist_ok=True)

    # Get the device
    device = torch.device("cuda")

    # Create the model
    logging.info(f"Creating the model...")
    if args.image_model == "label_only":
        label_map = utils.load_json(args.label_map_filename)
        model = clipsep.LabelSepV2(
            args.n_mix,
            args.n_labels,
            label_map,
            args.layers,
            args.channels,
            use_log_freq=args.log_freq,
            use_weighted_loss=args.weighted_loss,
            use_binary_mask=args.binary_mask,
        )
    elif args.image_model == "bert":
            label_map = utils.load_json(args.label_map_filename)
            model = clipsep.BERTSep(
                args.n_mix,
                label_map,
                args.layers,
                args.channels,
                use_log_freq=args.log_freq,
                use_weighted_loss=args.weighted_loss,
                use_binary_mask=args.binary_mask,
                bert_embeddings=args.bert_embeddings
            )
    elif args.image_model == "pit":
        model = clipsep.PITSep(
            args.n_mix,
            args.layers,
            args.channels,
            use_log_freq=args.log_freq,
            use_weighted_loss=args.weighted_loss,
            use_binary_mask=args.binary_mask,
        )
    elif args.image_model in ("clipsepnit", 'clippit4'):
        model = clipsep.CLIPPITSepV4(
            args.n_mix,
            args.layers,
            args.channels,
            use_log_freq=args.log_freq,
            use_weighted_loss=args.weighted_loss,
            use_binary_mask=args.binary_mask,
            reg_coef=args.reg_coef,
            reg2_coef=args.reg2_coef,
            reg2_epsilon=args.reg2_epsilon,
        )
    elif args.fusion == "late":
        model = clipsep.CLIPSep(
            args.n_mix,
            args.layers,
            args.channels,
            use_log_freq=args.log_freq,
            use_weighted_loss=args.weighted_loss,
            use_binary_mask=args.binary_mask,
        )
    elif args.fusion == "early":
        model = clipsep.CLIPSepV2(
            args.n_mix,
            args.layers,
            args.channels,
            use_log_freq=args.log_freq,
            use_weighted_loss=args.weighted_loss,
            use_binary_mask=args.binary_mask,
        )
    elif args.fusion == "middle":
        model = clipsep.CLIPSepV3(
            args.n_mix,
            args.layers,
            args.channels,
            use_log_freq=args.log_freq,
            use_weighted_loss=args.weighted_loss,
            use_binary_mask=args.binary_mask,
        )
    model = torch.nn.DataParallel(model, device_ids=range(args.gpus))
    model.to(device)

    # Create the image model
    if "clip" in args.image_model:
        # Load the pretrained CLIP net
        clip_net, _ = clip.load("ViT-B/32", device)
        clip_net.old_forward = clip_net.forward
        clip_net.forward = types.MethodType(new_clip_forward, clip_net)
        clip_net = torch.nn.DataParallel(clip_net, device_ids=range(args.gpus))
        clip_net.to(device)
        clip_net.eval()
    elif args.image_model == "sop":
        # Load the pretrained ResNet model
        res_net = clipsep.ResnetDilated(
            torchvision.models.resnet18(weights="DEFAULT")
        )
        res_net = torch.nn.DataParallel(res_net, device_ids=range(args.gpus))
        res_net.to(device)
        res_net.eval()

    # Load the checkpoint
    checkpoint_dir = args.out_dir / "checkpoints"
    if args.model_steps is None:
        checkpoint_filename = checkpoint_dir / "best_model.pt"
    else:
        checkpoint_filename = checkpoint_dir / f"model_{args.model_steps}.pt"
    model.load_state_dict(torch.load(checkpoint_filename, map_location=device))
    logging.info(f"Loaded the model weights from: {checkpoint_filename}")
    model.eval()

    # Dataset and loader
    test_dataset = dataset.MixDatasetV2(
        args.test_list,
        args.test_list2,
        "valid",
        audio_len=args.audio_len,
        audio_rate=args.audio_rate,
        n_fft=args.n_fft,
        hop_len=args.hop_len,
        win_len=args.win_len,
        n_frames=args.frames,
        stride_frames=args.stride_frames,
        img_size=args.img_size,
        fps=args.fps,
        preprocess_func=dataset.transform(),
        max_sample=args.n_visualization,
        return_waveform=False,
        audio_only=args.audio_only,
        normalize=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    # Star visualization
    logging.info("Visualizing...")

    # Switch to eval mode
    model.eval()

    # Open the HTML writers
    html_writers = {
        "text": visualization.HTMLWriter(args.vis_dir / "text" / "index.html"),
        "image": visualization.HTMLWriter(
            args.vis_dir / "image" / "index.html"
        ),
    }
    headers = get_html_headers(args.n_mix, args.image_model)
    html_writers["text"].write_header(headers)
    html_writers["image"].write_header(headers)

    # Start evaluation
    if args.audio_only or args.image_model in ("label_only", "bert", "pit"):
        test_modes = ["text"]
    elif args.image_model == "sop":
        test_modes = ["image"]
    else:
        test_modes = ["text", "image"]
    for mode in test_modes:
        with torch.no_grad():

            pbar = tqdm.tqdm(test_loader, ncols=120)
            for i, batch in enumerate(pbar):

                # Compute image embedding
                B = batch["mag_mix"].size(0)
                if "clip" in args.image_model:
                    if mode == "image":
                        frames = batch["frames"][0]
                        (B, T, C, H, W) = frames.size()
                        out = clip_net(image=frames.view(B * T, C, H, W)).type(
                            frames.dtype
                        )
                        C = out.size(1)
                        img_emb = out.view(B, T, C).mean(1)
                    elif mode == "text":
                        text_inputs = []
                        if args.prompt_ens:
                            for b in range(B):
                                prompts = get_text_prompts(batch["infos"][0][3][b])
                                text_inputs.append(torch.cat([clip.tokenize(prompt) for prompt in prompts]))
                            #text_inputs = torch.cat(text_inputs)
                            img_emb = torch.cat([clip_net(text=text_inp).type(
                                    batch["mag_mix"].dtype).mean(dim=0).unsqueeze(0)
                                 for text_inp in text_inputs])
                        else:
                            for b in range(B):
                                prompt = get_text_prompt(batch["infos"][0][3][b])
                                text_inputs.append(clip.tokenize(prompt))
                            text_inputs = torch.cat(text_inputs)
                            img_emb = clip_net(text=text_inputs).type(
                                batch["mag_mix"].dtype
                            )

                elif args.image_model == "sop":
                    frames = batch["frames"][0]
                    (B, T, C, H, W) = frames.size()
                    out = res_net(frames.view(B * T, C, H, W))
                    C = out.size(1)
                    img_emb = out.view(B, T, C).mean(1)
                else:
                    img_emb=[]

                batch["mag_mix"] = batch["mag_mix"].to(device)
                batch["mags"] = [x.to(device) for x in batch["mags"]]

                # Forward pass
                out = model.module.infer2(batch, [img_emb])

                # Output visualization
                output_visuals(
                    html_writers[mode],
                    batch,
                    out,
                    args.vis_dir / mode,
                    mode,
                    n_mix=args.n_mix,
                    n_fft=args.n_fft,
                    hop_len=args.hop_len,
                    win_len=args.win_len,
                    audio_rate=args.audio_rate,
                    n_frames=args.frames,
                    stride_frames=args.stride_frames,
                    fps=args.fps,
                    use_log_freq=args.log_freq,
                    use_binary_mask=args.binary,
                    image_model=args.image_model,
                )

    # Close the HTML writer
    html_writers["text"].close()
    html_writers["image"].close()


if __name__ == "__main__":
    # Parse command-lind arguments
    args = parse_args()

    # Make sure the output directory exists
    if args.vis_dir is None:
        args.vis_dir = args.out_dir / "visualization2"
    args.vis_dir.mkdir(exist_ok=True, parents=True)

    # Set up a console logger
    if args.log_filename is None:
        args.log_filename = args.vis_dir / "visualize.log"
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.log_filename, "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Save command-line arguments
    utils.save_args(args.vis_dir / "visualize-args.json", args)
    logging.info(f"Saved arguments to {args.vis_dir / 'visualize-args.json'}")

    # Load training configurations
    logging.info(
        f"Loading training arguments from: {args.out_dir / 'train-args.json'}"
    )
    train_args = utils.load_json(args.out_dir / "train-args.json")
    logging.info(f"Using loaded arguments:\n{pprint.pformat(train_args)}")
    for key in (
        "audio_len",
        "audio_rate",
        "n_fft",
        "hop_len",
        "win_len",
        "img_size",
        "fps",
        "n_mix",
        "fusion",
        "channels",
        "layers",
        "frames",
        "stride_frames",
        "binary_mask",
        "loss",
        "weighted_loss",
        "log_freq",
    ):
        setattr(args, key, train_args[key])

    # Handle backward compatibility
    args.image_model = train_args.get("image_model", "clip")
    args.train_mode = train_args.get("train_mode", "image")
    # args.audio_only = train_args.get("audio_only", False)
    args.n_labels = train_args.get("n_labels")
    args.label_map_filename = train_args.get("label_map_filename")
    args.reg_coef = train_args.get("reg_coef", 0)
    args.reg_epsilon = train_args.get("reg_epsilon", 0.1)
    args.reg2_coef = train_args.get("reg2_coef", 0)
    args.reg2_epsilon = train_args.get("reg2_epsilon", 0.5)

    # Run the main program
    main(args)
