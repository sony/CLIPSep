"""Evaluate the model."""
import argparse
import collections
import logging
import pathlib
import pprint
import random
import sys
import types

import clip
import mir_eval.separation
import museval.metrics
import numpy as np
import scipy.io.wavfile
import scipy.stats
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision
import tqdm

import clipsep
import dataset
import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
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
        help="filename of the test list 2",
    )
    parser.add_argument(
        "-n_eval",
        "--n_evaluation",
        type=int,
        help="number of samples to evaluate",
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
        "--backend",
        default="museval",
        choices=("museval", "mir_eval"),
        help="backend package used for evaluation",
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
        "--metrics",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="whether to compute metrics",
    )
    parser.add_argument(
        "--pit",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="whether to include PIT streams",
    )
    parser.add_argument(
        "--prompt_ens",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="whether to ensemble prompts",
    )
    parser.add_argument(
        "--bert_embeddings",
        type=pathlib.Path,
        help="filename of the bert embedding dictionary",
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
    parser.add_argument(
        "--n_test_sources",
        default=2,
        type=int,
        help="number of sources in the mixture. n>1 will be sampled from test list 2",
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
    return [f"a photo of {label}.", f"a photo of the small {label}.", f"a photo of the small {label}.", f"a low resolution photo of a {label}.", f"a photo of many {label}."]


def count_parameters(net):
    """Return the number of parameters in a model."""
    return sum(p.numel() for p in net.parameters())


def calc_metrics(
    batch,
    out,
    n_mix=2,
    n_fft=1024,
    hop_len=256,
    win_len=1024,
    use_log_freq=True,
    use_binary_mask=True,
    backend="mus_eval",
    threshold=0.5,
    image_model="clip",
    include_pit=True,
):
    """Calculate the evaluation metrics."""
    if "pit" in image_model:
        assert (
            n_mix == 2
        ), f"`n_mix` must be 2 when `image_model` is {image_model}"

    # Initialize lists
    metrics = collections.defaultdict(list)

    # Fetch data and predictions
    mag_mix = batch["mag_mix"]
    phase_mix = batch["phase_mix"]
    audios = batch["audios"]

    if "clipsepnit" in image_model:
        if image_model in ("clipsepnit"):
            if include_pit:
                pred_masks = [
                    torch.clamp(
                        torch.sigmoid(out["pred_masks"][0])
                        + torch.sigmoid(out["pit_masks"][0]),
                        max=1,
                    ),
                    1
                    - torch.clamp(
                        torch.sigmoid(out["pred_masks"][0])
                        + torch.sigmoid(out["pit_masks"][0]),
                        max=1,
                    ),
                ]
                pred_masks_alt = [
                    torch.clamp(
                        torch.sigmoid(out["pred_masks"][0])
                        + torch.sigmoid(out["pit_masks"][1]),
                        max=1,
                    ),
                    1
                    - torch.clamp(
                        torch.sigmoid(out["pred_masks"][0])
                        + torch.sigmoid(out["pit_masks"][1]),
                        max=1,
                    ),
                ]
            else:
                pred_masks = [
                    torch.sigmoid(out["pred_masks"][0]),
                    1 - torch.sigmoid(out["pred_masks"][0]),
                ]
        else:
            if include_pit:
                pred_masks = [
                    torch.sigmoid(out["pred_masks"][0] + out["pit_masks"][0]),
                    1
                    - torch.sigmoid(
                        out["pred_masks"][0] + out["pit_masks"][0]
                    ),
                ]
                pred_masks_alt = [
                    torch.sigmoid(out["pred_masks"][0] + out["pit_masks"][1]),
                    1
                    - torch.sigmoid(
                        out["pred_masks"][0] + out["pit_masks"][1]
                    ),
                ]
            else:
                pred_masks = [
                    torch.sigmoid(out["pred_masks"][0]),
                    1 - torch.sigmoid(out["pred_masks"][0]),
                ]
    elif image_model == "pit":
        pred_masks = [out["pred_masks"][0], 1 - out["pred_masks"][0]]
        pred_masks_alt = [1 - out["pred_masks"][0], out["pred_masks"][0]]
    else:
        pred_masks = [out["pred_masks"][0], 1 - out["pred_masks"][0]]

    # Unwarp log scale
    N = n_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None] * N
    for n in range(N):
        if use_log_freq:
            grid_unwarp = torch.from_numpy(
                utils.warpgrid(
                    B,
                    n_fft // 2 + 1,
                    pred_masks[0].size(3),
                    warp=False,
                )
            ).to(pred_masks[n].device)
            pred_masks_linear[n] = F.grid_sample(
                pred_masks[n], grid_unwarp, align_corners=True
            )
        else:
            pred_masks_linear[n] = pred_masks[n]
    if "pit" in image_model and include_pit:
        pred_masks_linear_alt = [None] * N
        for n in range(N):
            if use_log_freq:
                grid_unwarp_alt = torch.from_numpy(
                    utils.warpgrid(
                        B,
                        n_fft // 2 + 1,
                        pred_masks_alt[0].size(3),
                        warp=False,
                    )
                ).to(pred_masks_alt[n].device)
                pred_masks_linear_alt[n] = F.grid_sample(
                    pred_masks_alt[n], grid_unwarp_alt, align_corners=True
                )
            else:
                pred_masks_linear_alt[n] = pred_masks_alt[n]

    # Convert into numpy arrays
    mag_mix = mag_mix.detach().cpu().numpy()
    phase_mix = phase_mix.detach().cpu().numpy()
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()

        # Apply a threshold
        if use_binary_mask:
            pred_masks_linear[n] = (pred_masks_linear[n] > threshold).astype(
                np.float32
            )
    if "pit" in image_model and include_pit:
        for n in range(N):
            pred_masks_linear_alt[n] = (
                pred_masks_linear_alt[n].detach().cpu().numpy()
            )

            # Apply a threshold
            if use_binary_mask:
                pred_masks_linear_alt[n] = (
                    pred_masks_linear_alt[n] > threshold
                ).astype(np.float32)

    # Loop over each sample
    for j in range(B):

        # Reconstruct the mixture
        mix_wav = utils.istft_reconstruction(
            mag_mix[j, 0], phase_mix[j, 0], hop_len=hop_len, win_len=win_len
        )

        # Reconstruct each component
        pred_wavs = [None] * N
        for n in range(N):
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            pred_wavs[n] = utils.istft_reconstruction(
                pred_mag, phase_mix[j, 0], hop_len=hop_len, win_len=win_len
            )
        if "pit" in image_model and include_pit:
            pred_wavs_alt = [None] * N
            for n in range(N):
                pred_mag_alt = mag_mix[j, 0] * pred_masks_linear_alt[n][j, 0]
                pred_wavs_alt[n] = utils.istft_reconstruction(
                    pred_mag_alt,
                    phase_mix[j, 0],
                    hop_len=hop_len,
                    win_len=win_len,
                )

        # Compute separation performance
        L = pred_wavs[0].shape[0]
        #print('L = {}'.format(L))
        gts_wav = [None] * N
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(pred_wavs[n])) > 1e-5
        if not valid:
            continue
        # print(len(pred_wavs), pred_wavs[0].shape)
        if backend == "museval":
            sdr, _, sir, sar, _ = museval.metrics.bss_eval(
                np.asarray(gts_wav), np.asarray(pred_wavs), np.inf
            )
            sdr = sdr[:1, 0]
            sir = sir[:1, 0]
            sar = sar[:1, 0]
            (sdr_mix, _, sir_mix, sar_mix, _,) = museval.metrics.bss_eval(
                np.asarray(gts_wav), np.asarray([mix_wav[0:L]] * N), np.inf
            )
            sdr_mix = sdr_mix[:1, 0]
            sir_mix = sir_mix[:1, 0]
            sar_mix = sar_mix[:1, 0]
        elif backend == "mir_eval":
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                np.asarray(gts_wav), np.asarray(pred_wavs), False
            )
            (
                sdr_mix,
                sir_mix,
                sar_mix,
                _,
            ) = mir_eval.separation.bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mix_wav[0:L]] * N),
                False,
            )
        else:
            raise ValueError(f"Unknown backend : {backend}.")

        # Handle PIT
        if "pit" in image_model and include_pit:
            L = pred_wavs_alt[0].shape[0]
            valid = True
            for n in range(N):
                valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
                valid *= np.sum(np.abs(pred_wavs_alt[n])) > 1e-5
            if not valid:
                continue
            if backend == "museval":
                sdr_alt, _, sir_alt, sar_alt, _ = museval.metrics.bss_eval(
                    np.asarray(gts_wav), np.asarray(pred_wavs_alt), np.inf
                )
                sdr_alt = sdr_alt[:1, 0]
                sir_alt = sir_alt[:1, 0]
                sar_alt = sar_alt[:1, 0]
            elif backend == "mir_eval":
                (
                    sdr_alt,
                    sir_alt,
                    sar_alt,
                    _,
                ) = mir_eval.separation.bss_eval_sources(
                    np.asarray(gts_wav), np.asarray(pred_wavs_alt), False
                )
            else:
                raise ValueError(f"Unknown backend : {backend}.")

            if sdr.mean() >= sdr_alt.mean():
                metrics["sdr"].extend(sdr.tolist())
                metrics["sir"].extend(sir.tolist())
                metrics["sar"].extend(sar.tolist())
            else:
                metrics["sdr"].extend(sdr_alt.tolist())
                metrics["sir"].extend(sir_alt.tolist())
                metrics["sar"].extend(sar_alt.tolist())

        else:
            metrics["sdr"].extend(sdr.tolist())
            metrics["sir"].extend(sir.tolist())
            metrics["sar"].extend(sar.tolist())

        metrics["sdr_mix"].extend(sdr_mix.tolist())
        metrics["sir_mix"].extend(sir_mix.tolist())
        metrics["sar_mix"].extend(sar_mix.tolist())

    return metrics


def new_clip_forward(self, image=None, text=None):
    """A CLIP forward function that automatically chooses the mode."""
    if image is None and text is None:
        raise ValueError("Either `image` or `text` must be given.")
    if image is None:
        return self.encode_text(text)
    if text is None:
        return self.encode_image(image)
    return self.old_forward(image, text)


def main(args):
    """Main function."""
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    elif args.image_model == "clipsepnit":
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

    # Switch to eval mode
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
        max_sample=args.n_evaluation,
        return_waveform=False,
        audio_only=args.audio_only,
        N_test_sources=args.n_test_sources
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    # Star evaluation
    logging.info("Evaluating...")
    if args.audio_only or args.image_model in ("label_only", "bert", "pit"):
        test_modes = ["text"]
    elif args.image_model == "sop":
        test_modes = ["image"]
    else:
        test_modes = ["text", "image"]
    for mode in test_modes:
        # Initialize counters
        metrics = collections.defaultdict(list)

        # Iterate over the dataset
        pbar = tqdm.tqdm(test_loader, ncols=120)
        for batch in pbar:

            with torch.no_grad():
                # Compute image embedding
                B = batch["mag_mix"].size(0)
                img_emb = []
                if "clip" in args.image_model:
                    if mode == "image":
                        frames = batch["frames"][0]
                        (B, T, C, H, W) = frames.size()
                        out = clip_net(image=frames.view(B * T, C, H, W)).type(
                            frames.dtype
                        )
                        C = out.size(1)
                        img_emb = [out.view(B, T, C).mean(1)]
                    elif mode == "text":
                        text_inputs = []
                        if args.prompt_ens:
                            for b in range(B):
                                prompts = get_text_prompts(batch["infos"][0][3][b])
                                text_inputs.append(torch.cat([clip.tokenize(prompt) for prompt in prompts]))
                            #text_inputs = torch.cat(text_inputs)
                            img_emb = [
                                torch.cat([clip_net(text=text_inp).type(
                                    batch["mag_mix"].dtype).mean(dim=0).unsqueeze(0)
                                 for text_inp in text_inputs])
                            ]
                        else:
                            for b in range(B):
                                prompt = get_text_prompt(batch["infos"][0][3][b])
                                text_inputs.append(clip.tokenize(prompt))
                            text_inputs = torch.cat(text_inputs)
                            img_emb = [
                                clip_net(text=text_inputs).type(
                                    batch["mag_mix"].dtype
                                )
                            ]
                            

                elif args.image_model == "sop":
                    frames = batch["frames"][0]
                    (B, T, C, H, W) = frames.size()
                    out = res_net(frames.view(B * T, C, H, W))
                    C = out.size(1)
                    img_emb = [out.view(B, T, C).mean(1)]

                batch["mag_mix"] = batch["mag_mix"].to(device)
                batch["mags"] = [x.to(device) for x in batch["mags"]]

                # Forward pass
                out = model.module.infer2(batch, img_emb)

                # Calculate metrics
                batch_metrics = calc_metrics(
                    batch,
                    out,
                    n_mix=args.n_mix,
                    n_fft=args.n_fft,
                    hop_len=args.hop_len,
                    win_len=args.win_len,
                    use_log_freq=args.log_freq,
                    use_binary_mask=args.binary,
                    backend=args.backend,
                    threshold=args.threshold,
                    image_model=args.image_model,
                    include_pit=args.pit,
                )
                for key in batch_metrics:
                    metrics[key].extend(batch_metrics[key])

                pbar.set_postfix(
                    sdr=f"{np.mean(batch_metrics['sdr']):.2f}",
                    sir=f"{np.mean(batch_metrics['sir']):.2f}",
                    sar=f"{np.mean(batch_metrics['sar']):.2f}",
                )

        means = {key: np.mean(metrics[key]) for key in metrics}
        errs = {key: scipy.stats.sem(metrics[key]) for key in metrics}
        medians = {key: np.median(metrics[key]) for key in metrics}
        logging.info(
            f"Evaluation results ({mode} query): \n"
            f"sdr={means['sdr']:.4f}±{errs['sdr']:.4f}, "
            f"sir={means['sir']:.4f}±{errs['sir']:.4f}, "
            f"sar={means['sar']:.4f}±{errs['sar']:.4f}\n"
            f"sdr_median={medians['sdr']:.4f}, "
            f"sir_median={medians['sir']:.4f}, "
            f"sar_median={medians['sar']:.4f}\n"
            f"sdr_mix={means['sdr_mix']:.4f}±{errs['sdr_mix']:.4f}, "
            f"sir_mix={means['sir_mix']:.4f}±{errs['sir_mix']:.4f}, "
            f"sar_mix={means['sar_mix']:.4f}±{errs['sar_mix']:.4f}\n"
            f"sdr_mix_median={medians['sdr_mix']:.4f}, "
            f"sir_mix_median={medians['sir_mix']:.4f}, "
            f"sar_mix_median={medians['sar_mix']:.4f}"
        )


if __name__ == "__main__":
    # Parse command-lind arguments
    args = parse_args()

    # Make sure the output directory exists
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Set up a console logger
    if args.log_filename is None:
        args.log_filename = args.out_dir / "evaluate2.log"
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
    utils.save_args(args.out_dir / "evaluate2-args.json", args)
    logging.info(f"Saved arguments to {args.out_dir / 'evaluate2-args.json'}")

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
    args.bert_embeddings = train_args.get("bert_embeddings")
    args.reg_coef = train_args.get("reg_coef", 0)
    args.reg_epsilon = train_args.get("reg_epsilon", 0.1)
    args.reg2_coef = train_args.get("reg2_coef", 0)
    args.reg2_epsilon = train_args.get("reg2_epsilon", 0.5)

    # Run the main program
    main(args)
