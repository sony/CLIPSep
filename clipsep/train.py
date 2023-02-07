"""Train the model."""
import argparse
import logging
import pathlib
import pprint
import random
import shutil
import sys
import types

import clip
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision.models
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
        "--train_list",
        type=pathlib.Path,
        help="filename of the training list",
    )
    parser.add_argument(
        "-v",
        "--val_list",
        type=pathlib.Path,
        help="filename of the validation list",
    )
    parser.add_argument(
        "-n_val",
        "--n_validation",
        type=int,
        help="number of samples to evaluate",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=pathlib.Path,
        help="filename of the pretrained weights",
    )

    # Data
    parser.add_argument(
        "--batch_size", default=32, type=int, help="batch size"
    )
    parser.add_argument(
        "--drop_closest",
        type=int,
        help="number of the closest pairs to drop (-1 for soft dropping)",
    )
    parser.add_argument(
        "--drop_closest_steps",
        default=10000,
        type=int,
        help="when to start dropping closest pairs",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        help="repeat the dataset to avoid frequent reinitialization",
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

    # Audio
    parser.add_argument(
        "--audio_len",
        default=65535,
        type=int,
        help="audio length (in samples)",
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

    # Image
    parser.add_argument(
        "--img_size", default=224, type=int, help="size of input frame"
    )
    parser.add_argument(
        "--fps", default=1, type=float, help="video frame sampling rate"
    )

    # Model
    parser.add_argument(
        "--train_mode",
        default="image",
        choices=("image", "text", "hybrid"),
        help="training mode",
    )
    parser.add_argument(
        "--image_model",
        default="clip",
        choices=(
            "clip",
            "sop",
            "label_only",
            "bert",
            "pit",
            "clipsepnit",
        ),
        help="image model",
    )
    parser.add_argument(
        "--n_mix", default=2, type=int, help="number of sounds to mix"
    )
    parser.add_argument(
        "--fusion",
        default="late",
        choices=("early", "late", "middle"),
        help="fusion type",
    )
    parser.add_argument(
        "--channels", default=32, type=int, help="number of channels"
    )
    parser.add_argument(
        "--layers", default=7, type=int, help="number of U-Net layers (> 5)"
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
        "--binary_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use binary masks",
    )
    parser.add_argument("--loss", default="bce", help="loss function")
    parser.add_argument(
        "--weighted_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use weighted loss",
    )
    parser.add_argument(
        "--log_freq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use log frequency scale",
    )
    parser.add_argument("--n_labels", type=int, help="number of labels")
    parser.add_argument(
        "--label_map_filename",
        type=pathlib.Path,
        help="filename to the label map",
    )
    parser.add_argument(
        "--bert_embeddings",
        type=pathlib.Path,
        help="filename of the bert embedding dictionary",
    )
    parser.add_argument(
        "--reg_coef", default=0, type=float, help="regularizer weight"
    )
    parser.add_argument(
        "--reg_epsilon", default=0.1, type=float, help="regularizer epsilon"
    )
    parser.add_argument(
        "--reg2_coef", default=0.1, type=float, help="regularizer weight"
    )
    parser.add_argument(
        "--reg2_epsilon", default=0.25, type=float, help="regularizer epsilon"
    )

    # Training
    parser.add_argument(
        "--steps", default=200000, type=int, help="number of steps"
    )
    parser.add_argument(
        "--valid_steps", default=10000, type=int, help="validation frequency"
    )
    parser.add_argument(
        "--lr", default=0.001, type=float, help="learning rate"
    )
    parser.add_argument(
        "--lr_warmup_steps",
        default=5000,
        type=int,
        help="learning rate warmup steps",
    )
    parser.add_argument(
        "--lr_decay_steps",
        default=100000,
        type=int,
        help="learning rate decay end steps",
    )
    parser.add_argument(
        "--lr_decay_multiplier",
        default=0.1,
        type=float,
        help="learning rate multiplier at the end",
    )
    parser.add_argument(
        "--grad_norm_clip",
        default=1.0,
        type=float,
        help="gradient norm clipping",
    )
    parser.add_argument(
        "--pit_warmup_steps",
        default=0,
        type=int,
        help="pit stream warmup steps",
    )

    # Others
    parser.add_argument("--seed", default=1234, type=int, help="manual seed")
    parser.add_argument(
        "--gpus", default=1, type=int, help="number of gpus to use"
    )
    parser.add_argument(
        "--workers",
        default=8,
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


def count_parameters(net):
    """Return the number of parameters in a model."""
    return sum(p.numel() for p in net.parameters())


def get_lr_multiplier(
    step, warmup_steps, decay_end_steps, decay_end_multiplier
):
    """Return the learning rate multiplier with a warmup and decay schedule.
    The learning rate multiplier starts from 0 and linearly increases to 1
    after `warmup_steps`. After that, it linearly decreases to
    `decay_end_multiplier` until `decay_end_steps` is reached.
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step > decay_end_steps:
        return decay_end_multiplier
    position = (step - warmup_steps) / (decay_end_steps - warmup_steps)
    return 1 - (1 - decay_end_multiplier) * position


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

    # Make sure the checkpoint and sample directories exist
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)
    (args.out_dir / "samples").mkdir(exist_ok=True)
    (args.out_dir / "samples" / "text").mkdir(exist_ok=True)
    (args.out_dir / "samples" / "image").mkdir(exist_ok=True)

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
    logging.info(f"Total number of parameters: {count_parameters(model)}")

    # Load the pretrained weights
    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights, map_location=device))
        logging.info(f"Loaded the model weights from: {args.weights}")

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

    # Datasets and loaders
    logging.info("Creating the data loaders...")
    train_dataset = dataset.MixDataset(
        args.train_list,
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
        preprocess_func=dataset.transform(),
        max_sample=None,
        return_waveform=False,
        repeat=args.repeat,
        frame_margin=args.frame_margin,
        audio_only=args.audio_only,
    )
    if args.repeat is None:
        logging.info(f"Training set size: {len(train_dataset)}")
    else:
        logging.info(f"Training set size: {len(train_dataset) // args.repeat}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    val_dataset = dataset.MixDataset(
        args.val_list,
        "valid",
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
        preprocess_func=dataset.transform(),
        max_sample=args.n_validation,
        return_waveform=False,
        audio_only=args.audio_only,
    )
    logging.info(f"Validation set size: {len(val_dataset)}")
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_multiplier(
            step,
            args.lr_warmup_steps,
            args.lr_decay_steps,
            args.lr_decay_multiplier,
        ),
    )

    # Create a file to record losses
    loss_history = []
    if args.audio_only or "clip" in args.image_model:
        loss_header = "step,train_loss,val_loss"
    else:
        loss_header = "step,train_loss,val_loss_text,val_loss_img"

    if "clipsepnit" in args.image_model:
        act_history = []
        act_header = "step,mean_act,mean_pit_act"

    # Initialize variables
    step = 0
    min_val_loss = float("inf")

    # Iterate for the specified number of steps
    train_iterator = iter(train_loader)
    while step < args.steps:

        if (
            args.drop_closest is not None
            and args.drop_closest > 0
            and step > args.drop_closest_steps
        ):
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size + args.drop_closest,
                shuffle=True,
                num_workers=args.workers,
                drop_last=True,
            )
            train_iterator = iter(train_loader)

        # === Training ===

        logging.info("Training...")

        # Switch to training mode
        model.train()

        # Initialize variables
        recent_losses = []

        # Train the network for the specified number of steps
        pbar = tqdm.tqdm(range(args.valid_steps), ncols=120)
        for _ in pbar:

            # Get next batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                # Reinitialize dataset iterator
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            # Compute image embedding
            with torch.no_grad():
                img_emb = []
                for n in range(args.n_mix):
                    if "clip" in args.image_model:
                        # Set the training mode
                        if args.train_mode == "hybrid":
                            # Switch between image and text mode
                            if step % 2 == 0:
                                train_mode = "image"
                            else:
                                train_mode = "text"
                        else:
                            train_mode = args.train_mode
                        # Use the corresponding inputs
                        if train_mode == "image":
                            frames = batch["frames"][n]
                            (B, T, C, H, W) = frames.size()
                            out = clip_net(
                                image=frames.view(B * T, C, H, W)
                            ).type(frames.dtype)
                            C = out.size(1)
                            img_emb.append(out.view(B, T, C).mean(1))
                        elif train_mode == "text":
                            B = batch["mag_mix"].size(0)
                            text_inputs = []
                            for b in range(B):
                                prompt = get_text_prompt(
                                    batch["infos"][n][3][b]
                                )
                                text_inputs.append(clip.tokenize(prompt))
                            text_inputs = torch.cat(text_inputs)
                            img_emb.append(
                                clip_net(text=text_inputs).type(
                                    batch["mag_mix"].dtype
                                )
                            )

                    elif args.image_model == "sop":
                        frames = batch["frames"][n]
                        (B, T, C, H, W) = frames.size()
                        out = res_net(frames.view(B * T, C, H, W))
                        C = out.size(1)
                        img_emb.append(out.view(B, T, C).mean(1))

            # Forward pass
            optimizer.zero_grad()
            if step > args.drop_closest_steps:
                loss, out = model.forward(
                    batch, img_emb, drop_closest=args.drop_closest
                )
            elif (
                args.image_model == "clipsepnit" and step < args.pit_warmup_steps
            ):
                # Suppress the PIT loss during the warmup stage
                loss, out = model.forward(batch, img_emb, pit_loss=False)
            else:
                loss, out = model.forward(batch, img_emb)
            loss = loss.mean()

            if "clipsepnit" in args.image_model:
                mean_act = out["mean_act"]
                mean_pit_act = out["mean_pit_act"]

            # Backward pass
            loss.backward()
            if args.grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_norm_clip
                )
            optimizer.step()
            scheduler.step()

            # Compute the moving average for the loss
            recent_losses.append(float(loss))
            if len(recent_losses) > 100:
                del recent_losses[0]
            train_loss = np.mean(recent_losses)
            if "clipsepnit" in args.image_model:
                pbar.set_postfix(
                    loss=f"{train_loss:.4f}",
                    mean_act=f"{mean_act:.4f}",
                    mean_pit_act=f"{mean_pit_act:.4f}",
                )
            else:
                pbar.set_postfix(loss=f"{train_loss:.4f}")

            # Increment the step counter
            step += 1

        # === Validation ===

        logging.info("Validating...")

        # Switch to eval mode
        model.eval()

        # Start evaluation
        val_losses = {}
        if args.audio_only or args.image_model in ("label_only","bert", "pit"):
            val_modes = ["text"]
        elif args.image_model == "sop":
            val_modes = ["image"]
        else:
            val_modes = ["text", "image"]
        for mode in val_modes:
            with torch.no_grad():
                total_loss = 0
                count = 0

                if "clipsepnit" in args.image_model:
                    total_mean_act = 0
                    total_mean_pit_act = 0

                pbar = tqdm.tqdm(val_loader, ncols=120)
                for i, batch in enumerate(pbar):

                    # Compute image embedding
                    img_emb = []
                    for n in range(args.n_mix):
                        if "clip" in args.image_model:
                            if mode == "image":
                                frames = batch["frames"][n]
                                (B, T, C, H, W) = frames.size()
                                reshaped = frames.view(B * T, C, H, W)
                                out = clip_net(image=reshaped).type(
                                    frames.dtype
                                )
                                C = out.size(1)
                                img_emb.append(out.view(B, T, C).mean(1))
                            elif mode == "text":
                                B = batch["mag_mix"].size(0)
                                text_inputs = []
                                for b in range(B):
                                    prompt = get_text_prompt(
                                        batch["infos"][n][3][b]
                                    )
                                    text_inputs.append(clip.tokenize(prompt))
                                text_inputs = torch.cat(text_inputs)
                                img_emb.append(
                                    clip_net(text=text_inputs).type(
                                        batch["mag_mix"].dtype
                                    )
                                )
                        elif args.image_model == "sop":
                            frames = batch["frames"][n]
                            (B, T, C, H, W) = frames.size()
                            out = res_net(frames.view(B * T, C, H, W))
                            C = out.size(1)
                            img_emb.append(out.view(B, T, C).mean(1))

                    # Forward pass
                    loss, out = model.forward(batch, img_emb)
                    pbar.set_postfix(loss=f"{loss:.4f}")

                    # Accumulate loss
                    B = batch["mag_mix"].size(0)
                    total_loss += B * float(loss)
                    count += B

                    # Accumulate activation
                    if "clipsepnit" in args.image_model:
                        total_mean_act += B * float(out["mean_act"])
                        total_mean_pit_act += B * float(out["mean_pit_act"])

                # Log the average validation loss
                val_loss = total_loss / count
                val_losses[mode] = val_loss
                logging.info(
                    f"Validation loss ({mode} query) at step {step}: "
                    f"{val_loss:.4f}"
                )

                # Log the average activation
                if "clipsepnit" in args.image_model:
                    mean_act = total_mean_act / count
                    mean_pit_act = total_mean_pit_act / count
                    logging.info(
                        f"Activations: mean_act={mean_act:.4f}, "
                        f"mean_pit_act={mean_pit_act:.4f}"
                    )

        # Write losses to file
        if args.audio_only or args.image_model in ("label_only", "bert", "pit"):
            loss_history.append((step, train_loss, val_losses["text"]))
        elif args.image_model == "sop":
            loss_history.append((step, train_loss, val_losses["image"]))
        else:
            loss_history.append(
                (step, train_loss, val_losses["text"], val_losses["image"])
            )
        utils.save_csv(
            args.out_dir / "loss.csv",
            loss_history,
            fmt="%f",
            header=loss_header,
        )

        # Write activations to file
        if "clipsepnit" in args.image_model:
            act_history.append((step, mean_act, mean_pit_act))
            utils.save_csv(
                args.out_dir / "act.csv",
                act_history,
                fmt="%f",
                header=act_header,
            )

        # Save the model
        checkpoint_filename = args.out_dir / "checkpoints" / f"model_{step}.pt"
        torch.save(model.state_dict(), checkpoint_filename)
        logging.info(f"Saved the model to: {checkpoint_filename}")

        # Copy the model if it is the best model so far
        if args.image_model in ("label_only", "bert", "pit"):
            val_mode = "text"
        elif args.train_mode == "hybrid":
            val_mode = "image"
        else:
            val_mode = args.train_mode
        if val_losses[val_mode] < min_val_loss:
            min_val_loss = val_losses[val_mode]
            shutil.copyfile(
                checkpoint_filename,
                args.out_dir / "checkpoints" / "best_model.pt",
            )

    # Log minimum validation loss
    logging.info(f"Minimum validation loss achieved: {min_val_loss:.4f}")

    # Save the optimizer states
    optimizer_filename = args.out_dir / "checkpoints" / f"optimizer_{step}.pt"
    torch.save(optimizer.state_dict(), optimizer_filename)
    logging.info(f"Saved the optimizer state to: {optimizer_filename}")

    # Save the scheduler states
    scheduler_filename = args.out_dir / "checkpoints" / f"scheduler_{step}.pt"
    torch.save(scheduler.state_dict(), scheduler_filename)
    logging.info(f"Saved the scheduler state to: {scheduler_filename}")


if __name__ == "__main__":
    # Parse command-lind arguments
    args = parse_args()

    # Make sure the output directory exists
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Set up a console logger
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.out_dir / "train.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Save command-line arguments
    utils.save_args(args.out_dir / "train-args.json", args)
    logging.info(f"Saved arguments to {args.out_dir / 'train-args.json'}")

    # Run the main program
    main(args)
