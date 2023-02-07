"""Define the models."""
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


def init_weights(net):
    classname = net.__class__.__name__
    if classname.find("Conv") != -1:
        net.weight.data.normal_(0.0, 0.001)
    elif classname.find("BatchNorm") != -1:
        net.weight.data.normal_(1.0, 0.02)
        net.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        net.weight.data.normal_(0.0, 0.0001)


class CLIPSep(torch.nn.Module):
    """Separation model based on the CLIP model."""

    def __init__(
        self,
        n_mix,
        layers=7,
        channels=32,
        use_log_freq=True,
        use_weighted_loss=True,
        use_binary_mask=True,
    ):
        super().__init__()
        self.n_mix = n_mix
        self.use_log_freq = use_log_freq
        self.use_weighted_loss = use_weighted_loss
        self.use_binary_mask = use_binary_mask

        # Create the neural net
        self.sound_net = UNet(in_dim=1, out_dim=channels, num_downs=layers)
        self.frame_net = nn.Linear(512, channels)
        self.synth_net = InnerProd(fc_dim=channels)

        # Initialize the weights
        self.sound_net.apply(init_weights)
        self.frame_net.apply(init_weights)
        self.synth_net.apply(init_weights)

    def forward(self, batch, img_emb, drop_closest=None):
        N = self.n_mix
        mag_mix = batch["mag_mix"]
        mags = batch["mags"]

        # Pass through the frame net -> Bx1xC
        feat_frames_pre = [self.frame_net(img_emb[n]) for n in range(N)]
        feat_frames = [torch.sigmoid(feat) for feat in feat_frames_pre]

        # Compute similarities
        if drop_closest is not None:
            assert N == 2, "N must be 2 when `drop_closest` is enabled."
            similarities = F.cosine_similarity(
                img_emb[0].detach(), img_emb[1].detach()
            )

        # Drop most similar pairs
        if drop_closest is not None and drop_closest > 0:
            # Sort the similarities
            sorted_indices = torch.argsort(similarities)

            # Keep only those with low similarities
            mag_mix = mag_mix[sorted_indices[:-drop_closest]]
            for n in range(N):
                mags[n] = mags[n][sorted_indices[:-drop_closest]]
                feat_frames[n] = feat_frames[n][sorted_indices[:-drop_closest]]

        mag_mix = mag_mix + 1e-10

        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # Warp the spectrogram
        if self.use_log_freq:
            grid_warp = torch.from_numpy(
                utils.warpgrid(B, 256, T, warp=True)
            ).to(mag_mix.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp, align_corners=True)

        # Calculate loss weighting coefficient (magnitude of input mixture)
        if self.use_weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # Drop most similar pairs
        if drop_closest is not None and drop_closest == -1:
            # Desired weight as a function of similarity:
            #   sim    -1 <-> 0.5 <---------------> 1
            #   weight  1      1    2 x (1 - sim)   0
            w = F.relu(1 - 2 * F.relu(similarities - 0.5))
            weight *= w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Compute ground truth masks after warping!
        gt_masks = [None] * N
        for n in range(N):
            if self.use_binary_mask:
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / sum(mags[n])
                gt_masks[n].clamp_(0.0, 1.0)

        # Compute log magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # Pass through the sound net -> BxCxHxW
        feat_sound = self.sound_net(log_mag_mix)

        # Pass through the synth net
        pred_masks = [
            self.synth_net(feat_frames[n], feat_sound) for n in range(N)
        ]

        # Activate with Sigmoid function if using binary mask
        if self.use_binary_mask:
            pred_masks = [torch.sigmoid(mask) for mask in pred_masks]

        # Compute the loss
        loss = torch.mean(
            torch.stack(
                [
                    F.binary_cross_entropy(pred_masks[n], gt_masks[n], weight)
                    for n in range(N)
                ]
            )
        )

        return (
            loss,
            {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "mag_mix": mag_mix,
                "mags": mags,
                "weight": weight,
            },
        )

    def infer(self, mag_mix, img_emb, n_mix=1):
        N = n_mix

        # Pass through the frame net -> Bx1xC
        feat_frames_pre = [self.frame_net(img_emb[n]) for n in range(N)]
        feat_frames = [torch.sigmoid(feat) for feat in feat_frames_pre]

        mag_mix = mag_mix + 1e-10

        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # Warp the spectrogram
        if self.use_log_freq:
            grid_warp = torch.from_numpy(
                utils.warpgrid(B, 256, T, warp=True)
            ).to(mag_mix.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)

        # Compute log magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # Pass through the sound net -> BxCxHxW
        feat_sound = self.sound_net(log_mag_mix)

        # Pass through the synth net
        pred_masks = [
            self.synth_net(feat_frames[n], feat_sound) for n in range(N)
        ]

        # Activate with Sigmoid function if using binary mask
        if self.use_binary_mask:
            pred_masks = [torch.sigmoid(mask) for mask in pred_masks]

        return pred_masks

    def infer2(self, batch, img_emb):
        N = self.n_mix
        mag_mix = batch["mag_mix"]
        mags = batch["mags"]

        # Pass through the frame net -> Bx1xC
        feat_frames_pre = [self.frame_net(img_emb[0])]
        feat_frames = [torch.sigmoid(feat) for feat in feat_frames_pre]

        mag_mix = mag_mix + 1e-10

        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # Warp the spectrogram
        if self.use_log_freq:
            grid_warp = torch.from_numpy(
                utils.warpgrid(B, 256, T, warp=True)
            ).to(mag_mix.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp, align_corners=True)

        # Calculate loss weighting coefficient (magnitude of input mixture)
        if self.use_weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # Compute ground truth masks after warping!
        gt_masks = [None] * N
        for n in range(N):
            if self.use_binary_mask:
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / sum(mags[n])
                gt_masks[n].clamp_(0.0, 1.0)

        # Compute log magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # Pass through the sound net -> BxCxHxW
        feat_sound = self.sound_net(log_mag_mix)

        # Pass through the synth net
        pred_masks = [self.synth_net(feat_frames[0], feat_sound)]

        # Activate with Sigmoid function if using binary mask
        if self.use_binary_mask:
            pred_masks = [torch.sigmoid(pred_masks[0])]

        return {
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
            "mag_mix": mag_mix,
            "mags": mags,
            "weight": weight,
        }

    def infer3(self, batch, img_emb):

        mag_mix = batch["mag_mix"]

        # Pass through the frame net -> Bx1xC
        feat_frames_pre = [self.frame_net(img_emb)]
        feat_frames = [torch.sigmoid(feat) for feat in feat_frames_pre]

        mag_mix = mag_mix + 1e-10

        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # Warp the spectrogram
        if self.use_log_freq:
            grid_warp = torch.from_numpy(
                utils.warpgrid(B, 256, T, warp=True)
            ).to(mag_mix.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)

        # Calculate loss weighting coefficient (magnitude of input mixture)
        if self.use_weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # Compute log magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # Pass through the sound net -> BxCxHxW
        feat_sound = self.sound_net(log_mag_mix)

        # Pass through the synth net
        pred_masks = [self.synth_net(feat_frames[0], feat_sound)]

        # Get the input to the PIT stream
        # mean_feat_frames_pre = feat_frames_pre[0]
        # feat_pit_pre = [net(mean_feat_frames_pre) for net in self.pit_nets]
        # feat_pit = [torch.sigmoid(feat) for feat in feat_pit_pre]

        # Pass through the synth net for the PIT stream
        # pit_masks = [self.synth_net(feat, feat_sound) for feat in feat_pit]

        # Mean activation
        mean_act = torch.mean(torch.sigmoid(pred_masks[0]))
        # mean_pit_act = torch.mean(
        #     torch.sigmoid(pit_masks[0]) + torch.sigmoid(pit_masks[1])
        # )

        return {
            "pred_masks": pred_masks,
            # "pit_masks": pit_masks,
            "mag_mix": mag_mix,
            "weight": weight,
            "mean_act": mean_act,
            # "mean_pit_act": mean_pit_act,
        }


class CLIPSepV2(torch.nn.Module):
    """Separation model based on the CLIP model."""

    def __init__(
        self,
        n_mix,
        layers=7,
        channels=32,
        use_log_freq=True,
        use_weighted_loss=True,
        use_binary_mask=True,
    ):
        super().__init__()
        self.channels = channels
        self.n_mix = n_mix
        self.use_log_freq = use_log_freq
        self.use_weighted_loss = use_weighted_loss
        self.use_binary_mask = use_binary_mask

        # Create the neural net
        self.sound_net = UNet(in_dim=1 + channels, out_dim=1, num_downs=layers)
        self.frame_net = FCPool(fc_dim=channels)

        # Initialize the weights
        self.sound_net.apply(init_weights)
        self.frame_net.apply(init_weights)

    def forward(self, batch, img_emb, drop_closest=None):
        N = self.n_mix
        mag_mix = batch["mag_mix"]
        mags = batch["mags"]

        # Pass through the frame net -> Bx1xC
        feat_frames = [
            torch.sigmoid(self.frame_net(img_emb[n])) for n in range(N)
        ]

        # Drop most similar pairs
        if drop_closest is not None and drop_closest > 0:
            assert N == 2, "N must be 2 when `drop_closest` is enabled."
            similarities = F.cosine_similarity(feat_frames[0], feat_frames[1])
            sorted_indices = torch.argsort(similarities)

            # Keep only those with low similarities
            mag_mix = mag_mix[sorted_indices[:-drop_closest]]
            for n in range(N):
                mags[n] = mags[n][sorted_indices[:-drop_closest]]
                feat_frames[n] = feat_frames[n][sorted_indices[:-drop_closest]]

        mag_mix = mag_mix + 1e-10

        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # Warp the spectrogram
        if self.use_log_freq:
            grid_warp = torch.from_numpy(
                utils.warpgrid(B, 256, T, warp=True)
            ).to(mag_mix.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp, align_corners=True)

        # Calculate loss weighting coefficient (magnitude of input mixture)
        if self.use_weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # Compute ground truth masks after warping!
        gt_masks = [None] * N
        for n in range(N):
            if self.use_binary_mask:
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / sum(mags[n])
                gt_masks[n].clamp_(0.0, 1.0)

        # Compute log magnitude
        log_mag_mix = torch.log(mag_mix).detach()
        _, _, H, W = log_mag_mix.size()

        # Fuse visual features
        pred_masks = []
        for n in range(N):
            condition = feat_frames[n].unsqueeze(-1).unsqueeze(
                -1
            ) * torch.ones((B, self.channels, H, W), device=mag_mix.device)
            concated = torch.concat((log_mag_mix, condition), 1)

            # Pass through the sound net -> BxCxHxW
            feat_sound = self.sound_net(concated)

            pred_masks.append(feat_sound)

        # Activate with Sigmoid function if using binary mask
        if self.use_binary_mask:
            pred_masks = [torch.sigmoid(mask) for mask in pred_masks]

        # Compute the loss
        loss = torch.mean(
            torch.stack(
                [
                    F.binary_cross_entropy(pred_masks[n], gt_masks[n], weight)
                    for n in range(N)
                ]
            )
        )

        return (
            loss,
            {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "mag_mix": mag_mix,
                "mags": mags,
                "weight": weight,
            },
        )


class CLIPSepV3(torch.nn.Module):
    """Separation model based on the CLIP model."""

    def __init__(
        self,
        n_mix,
        layers=7,
        channels=32,
        use_log_freq=True,
        use_weighted_loss=True,
        use_binary_mask=True,
    ):
        super().__init__()
        self.n_mix = n_mix
        self.use_log_freq = use_log_freq
        self.use_weighted_loss = use_weighted_loss
        self.use_binary_mask = use_binary_mask

        # Create the neural net
        self.sound_net = CondUNet(
            in_dim=1, out_dim=1, cond_dim=channels, num_downs=layers
        )
        self.frame_net = nn.Linear(512, channels)
        self.synth_net = InnerProd(fc_dim=channels)

        # Initialize the weights
        self.sound_net.apply(init_weights)
        self.frame_net.apply(init_weights)
        self.synth_net.apply(init_weights)

    def forward(self, batch, img_emb, drop_closest=None):
        N = self.n_mix
        mag_mix = batch["mag_mix"]
        mags = batch["mags"]

        # Pass through the frame net -> Bx1xC
        feat_frames = [
            torch.sigmoid(self.frame_net(img_emb[n])) for n in range(N)
        ]

        # Drop most similar pairs
        if drop_closest is not None and drop_closest > 0:
            assert N == 2, "N must be 2 when `drop_closest` is enabled."
            similarities = F.cosine_similarity(feat_frames[0], feat_frames[1])
            sorted_indices = torch.argsort(similarities)

            # Combine with some threshold
            # (1 - cos_sim) / 2 -> [0, 1]

            # Keep only those with low similarities
            mag_mix = mag_mix[sorted_indices[:-drop_closest]]
            for n in range(N):
                mags[n] = mags[n][sorted_indices[:-drop_closest]]
                feat_frames[n] = feat_frames[n][sorted_indices[:-drop_closest]]

        mag_mix = mag_mix + 1e-10

        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # Warp the spectrogram
        if self.use_log_freq:
            grid_warp = torch.from_numpy(
                utils.warpgrid(B, 256, T, warp=True)
            ).to(mag_mix.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp, align_corners=True)

        # Calculate loss weighting coefficient (magnitude of input mixture)
        if self.use_weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # Compute ground truth masks after warping!
        gt_masks = [None] * N
        for n in range(N):
            if self.use_binary_mask:
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / sum(mags[n])
                gt_masks[n].clamp_(0.0, 1.0)

        # Compute log magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # Fuse visual features
        pred_masks = []
        for n in range(N):
            # Pass through the sound net -> BxCxHxW
            feat_sound = self.sound_net(log_mag_mix, feat_frames[n])

            pred_masks.append(feat_sound)

        # Activate with Sigmoid function if using binary mask
        if self.use_binary_mask:
            pred_masks = [torch.sigmoid(mask) for mask in pred_masks]

        # Compute the loss
        loss = torch.mean(
            torch.stack(
                [
                    F.binary_cross_entropy(pred_masks[n], gt_masks[n], weight)
                    for n in range(N)
                ]
            )
        )

        return (
            loss,
            {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "mag_mix": mag_mix,
                "mags": mags,
                "weight": weight,
            },
        )



class LabelSepV2(torch.nn.Module):
    """Separation model that relies only on the labels."""

    def __init__(
        self,
        n_mix,
        n_labels,
        label_map,
        layers=7,
        channels=32,
        use_log_freq=True,
        use_weighted_loss=True,
        use_binary_mask=True,
    ):
        super().__init__()
        self.n_mix = n_mix
        self.n_labels = n_labels
        self.label_map = label_map
        self.use_log_freq = use_log_freq
        self.use_weighted_loss = use_weighted_loss
        self.use_binary_mask = use_binary_mask

        # Create the neural net
        self.sound_net = UNet(in_dim=1, out_dim=channels, num_downs=layers)
        self.label_net = nn.Embedding(n_labels, 512)
        self.frame_net = nn.Linear(512, channels)
        self.synth_net = InnerProd(fc_dim=channels)

        # Initialize the weights
        self.sound_net.apply(init_weights)
        self.frame_net.apply(init_weights)
        self.synth_net.apply(init_weights)

    def forward(self, batch, img_emb, drop_closest=None):
        assert not drop_closest, "drop_closest not implemented!"

        N = self.n_mix
        mag_mix = batch["mag_mix"]
        mags = batch["mags"]

        # Get the label embedding
        label_emb = [
            self.label_net(
                torch.tensor(
                    [self.label_map[label] for label in batch["infos"][n][3]],
                    dtype=torch.int,
                    device=mag_mix.device,
                )
            )
            for n in range(N)
        ]

        # Pass through the frame net -> Bx1xC
        feat_frames = [
            torch.sigmoid(self.frame_net(label_emb[n])) for n in range(N)
        ]

        mag_mix = mag_mix + 1e-10

        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # Warp the spectrogram
        if self.use_log_freq:
            grid_warp = torch.from_numpy(
                utils.warpgrid(B, 256, T, warp=True)
            ).to(mag_mix.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp, align_corners=True)

        # Calculate loss weighting coefficient (magnitude of input mixture)
        if self.use_weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # Compute ground truth masks after warping!
        gt_masks = [None] * N
        for n in range(N):
            if self.use_binary_mask:
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / sum(mags[n])
                gt_masks[n].clamp_(0.0, 1.0)

        # Compute log magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # Pass through the sound net -> BxCxHxW
        feat_sound = self.sound_net(log_mag_mix)

        # Pass through the synth net
        pred_masks = [
            self.synth_net(feat_frames[n], feat_sound) for n in range(N)
        ]

        # Activate with Sigmoid function if using binary mask
        if self.use_binary_mask:
            pred_masks = [torch.sigmoid(mask) for mask in pred_masks]

        # Compute the loss
        loss = torch.mean(
            torch.stack(
                [
                    F.binary_cross_entropy(pred_masks[n], gt_masks[n], weight)
                    for n in range(N)
                ]
            )
        )

        return (
            loss,
            {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "mag_mix": mag_mix,
                "mags": mags,
                "weight": weight,
            },
        )

    def infer2(self, batch, img_emb):
        N = self.n_mix
        mag_mix = batch["mag_mix"]
        mags = batch["mags"]

        # Get the label embedding
        label_emb = [
            self.label_net(
                torch.tensor(
                    [self.label_map[label] for label in batch["infos"][0][3]],
                    dtype=torch.int,
                    device=mag_mix.device,
                )
            )
        ]

        # Pass through the frame net -> Bx1xC
        feat_frames = [torch.sigmoid(self.frame_net(label_emb[0]))]

        mag_mix = mag_mix + 1e-10

        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # Warp the spectrogram
        if self.use_log_freq:
            grid_warp = torch.from_numpy(
                utils.warpgrid(B, 256, T, warp=True)
            ).to(mag_mix.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp, align_corners=True)

        # Calculate loss weighting coefficient (magnitude of input mixture)
        if self.use_weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # Compute ground truth masks after warping!
        gt_masks = [None] * N
        for n in range(N):
            if self.use_binary_mask:
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / sum(mags[n])
                gt_masks[n].clamp_(0.0, 1.0)

        # Compute log magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # Pass through the sound net -> BxCxHxW
        feat_sound = self.sound_net(log_mag_mix)

        # Pass through the synth net
        pred_masks = [self.synth_net(feat_frames[0], feat_sound)]

        # Activate with Sigmoid function if using binary mask
        if self.use_binary_mask:
            pred_masks = [torch.sigmoid(pred_masks[0])]

        return {
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
            "mag_mix": mag_mix,
            "mags": mags,
            "weight": weight,
        }


class BERTSep(torch.nn.Module):
    """Separation model that relies only on the labels."""

    def __init__(
        self,
        n_mix,
        label_map,
        layers=7,
        channels=32,
        use_log_freq=True,
        use_weighted_loss=True,
        use_binary_mask=True,
        bert_embeddings='bert_embeddings.pt'
    ):
        super().__init__()
        self.n_mix = n_mix
        self.label_map = label_map
        self.use_log_freq = use_log_freq
        self.use_weighted_loss = use_weighted_loss
        self.use_binary_mask = use_binary_mask

        # Create the neural net
        self.sound_net = UNet(in_dim=1, out_dim=channels, num_downs=layers)
        self.frame_net = nn.Linear(256, channels)
        self.synth_net = InnerProd(fc_dim=channels)

        self.bert_embeddings = torch.load(bert_embeddings)
        # self.bert_embeddings = {k: torch.nn.Parameter(v, requires_grad=False) for k,v in self.bert_embeddings.items()}

        # Initialize the weights
        self.sound_net.apply(init_weights)
        self.frame_net.apply(init_weights)
        self.synth_net.apply(init_weights)

    def forward(self, batch, img_emb, drop_closest=None):
        assert not drop_closest, "drop_closest not implemented!"

        N = self.n_mix
        mag_mix = batch["mag_mix"]
        mags = batch["mags"]

        # Get the label embedding
        label_emb = [
            torch.cat([self.bert_embeddings[label]  for label in batch["infos"][n][3]], dim=0).to(mag_mix.device)
            for n in range(N)
        ]
        # label_emb = [
        #     self.label_net(
        #         torch.tensor(
        #             [self.label_map[label] for label in batch["infos"][n][3]],
        #             dtype=torch.int,
        #             device=mag_mix.device,
        #         )
        #     )
        #     for n in range(N)
        # ]

        # Pass through the frame net -> Bx1xC
        feat_frames = [
            torch.sigmoid(self.frame_net(label_emb[n])) for n in range(N)
        ]

        mag_mix = mag_mix + 1e-10

        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # Warp the spectrogram
        if self.use_log_freq:
            grid_warp = torch.from_numpy(
                utils.warpgrid(B, 256, T, warp=True)
            ).to(mag_mix.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp, align_corners=True)

        # Calculate loss weighting coefficient (magnitude of input mixture)
        if self.use_weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # Compute ground truth masks after warping!
        gt_masks = [None] * N
        for n in range(N):
            if self.use_binary_mask:
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / sum(mags[n])
                gt_masks[n].clamp_(0.0, 1.0)

        # Compute log magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # Pass through the sound net -> BxCxHxW
        feat_sound = self.sound_net(log_mag_mix)

        # Pass through the synth net
        pred_masks = [
            self.synth_net(feat_frames[n], feat_sound) for n in range(N)
        ]

        # Activate with Sigmoid function if using binary mask
        if self.use_binary_mask:
            pred_masks = [torch.sigmoid(mask) for mask in pred_masks]

        # Compute the loss
        loss = torch.mean(
            torch.stack(
                [
                    F.binary_cross_entropy(pred_masks[n], gt_masks[n], weight)
                    for n in range(N)
                ]
            )
        )

        return (
            loss,
            {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "mag_mix": mag_mix,
                "mags": mags,
                "weight": weight,
            },
        )

    def infer2(self, batch, img_emb):
        N = self.n_mix
        mag_mix = batch["mag_mix"]
        mags = batch["mags"]

        # Get the label embedding
        label_emb = [
            torch.cat([self.bert_embeddings[label]  for label in batch["infos"][n][3]], dim=0).to(mag_mix.device)
            for n in range(N)
        ]
        # label_emb = [
        #     self.label_net(
        #         torch.tensor(
        #             [self.label_map[label] for label in batch["infos"][0][3]],
        #             dtype=torch.int,
        #             device=mag_mix.device,
        #         )
        #     )
        # ]

        # Pass through the frame net -> Bx1xC
        feat_frames = [torch.sigmoid(self.frame_net(label_emb[0]))]

        mag_mix = mag_mix + 1e-10

        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # Warp the spectrogram
        if self.use_log_freq:
            grid_warp = torch.from_numpy(
                utils.warpgrid(B, 256, T, warp=True)
            ).to(mag_mix.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp, align_corners=True)

        # Calculate loss weighting coefficient (magnitude of input mixture)
        if self.use_weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # Compute ground truth masks after warping!
        gt_masks = [None] * N
        for n in range(N):
            if self.use_binary_mask:
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / sum(mags[n])
                gt_masks[n].clamp_(0.0, 1.0)

        # Compute log magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # Pass through the sound net -> BxCxHxW
        feat_sound = self.sound_net(log_mag_mix)

        # Pass through the synth net
        pred_masks = [self.synth_net(feat_frames[0], feat_sound)]

        # Activate with Sigmoid function if using binary mask
        if self.use_binary_mask:
            pred_masks = [torch.sigmoid(pred_masks[0])]

        return {
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
            "mag_mix": mag_mix,
            "mags": mags,
            "weight": weight,
        }



class PITSep(torch.nn.Module):
    """Separation model that relies only on permutation invariant training."""

    def __init__(
        self,
        n_mix,
        layers=7,
        channels=32,
        use_log_freq=True,
        use_weighted_loss=True,
        use_binary_mask=True,
    ):
        assert n_mix == 2, "PITSep supports only N = 2 for now."
        super().__init__()
        self.n_mix = n_mix
        self.use_log_freq = use_log_freq
        self.use_weighted_loss = use_weighted_loss
        self.use_binary_mask = use_binary_mask

        # Create the neural net
        self.sound_net = UNet(in_dim=1, out_dim=channels, num_downs=layers)
        self.synth_net = nn.Conv2d(
            channels, 2, kernel_size=1, stride=1, padding=0
        )

        # Initialize the weights
        self.sound_net.apply(init_weights)
        self.synth_net.apply(init_weights)

    def forward(self, batch, img_emb, drop_closest=None):
        assert not drop_closest, "drop_closest not implemented!"

        N = self.n_mix
        mag_mix = batch["mag_mix"]
        mags = batch["mags"]

        mag_mix = mag_mix + 1e-10

        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # Warp the spectrogram
        if self.use_log_freq:
            grid_warp = torch.from_numpy(
                utils.warpgrid(B, 256, T, warp=True)
            ).to(mag_mix.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp, align_corners=True)

        # Calculate loss weighting coefficient (magnitude of input mixture)
        if self.use_weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # Compute ground truth masks after warping!
        gt_masks = [None] * N
        for n in range(N):
            if self.use_binary_mask:
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / sum(mags[n])
                gt_masks[n].clamp_(0.0, 1.0)

        # Compute log magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # Pass through the sound net -> BxCxHxW
        feat_sound = self.sound_net(log_mag_mix)

        # Pass through the synth net
        pred_masks = self.synth_net(feat_sound)
        pred_masks = [pred_masks[:, 0:1], pred_masks[:, 1:2]]

        # Activate with Sigmoid function if using binary mask
        if self.use_binary_mask:
            pred_masks = [torch.sigmoid(mask) for mask in pred_masks]

        # Compute the loss
        loss_1 = (
            torch.mean(
                F.binary_cross_entropy(
                    pred_masks[0], gt_masks[0], weight, reduction="none"
                ),
                (1, 2, 3),
            )
            + torch.mean(
                F.binary_cross_entropy(
                    pred_masks[1], gt_masks[1], weight, reduction="none"
                ),
                (1, 2, 3),
            )
        ) / 2
        loss_2 = (
            torch.mean(
                F.binary_cross_entropy(
                    pred_masks[0], gt_masks[1], weight, reduction="none"
                ),
                (1, 2, 3),
            )
            + torch.mean(
                F.binary_cross_entropy(
                    pred_masks[1], gt_masks[0], weight, reduction="none"
                ),
                (1, 2, 3),
            )
        ) / 2
        loss = torch.mean(torch.minimum(loss_1, loss_2))

        return (
            loss,
            {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "mag_mix": mag_mix,
                "mags": mags,
                "weight": weight,
            },
        )

    def infer2(self, batch, img_emb):
        N = self.n_mix
        mag_mix = batch["mag_mix"]
        mags = batch["mags"]

        mag_mix = mag_mix + 1e-10

        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # Warp the spectrogram
        if self.use_log_freq:
            grid_warp = torch.from_numpy(
                utils.warpgrid(B, 256, T, warp=True)
            ).to(mag_mix.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp, align_corners=True)

        # Calculate loss weighting coefficient (magnitude of input mixture)
        if self.use_weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # Compute ground truth masks after warping!
        gt_masks = [None] * N
        for n in range(N):
            if self.use_binary_mask:
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / sum(mags[n])
                gt_masks[n].clamp_(0.0, 1.0)

        # Compute log magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # Pass through the sound net -> BxCxHxW
        feat_sound = self.sound_net(log_mag_mix)

        # Pass through the synth net
        pred_masks = self.synth_net(feat_sound)
        pred_masks = [pred_masks[:, 0:1], pred_masks[:, 1:2]]

        # Activate with Sigmoid function if using binary mask
        if self.use_binary_mask:
            pred_masks = [torch.sigmoid(mask) for mask in pred_masks]

        return {
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
            "mag_mix": mag_mix,
            "mags": mags,
            "weight": weight,
        }


class CLIPPITSepV4(torch.nn.Module):
    """Separation model based on the CLIP model."""

    def __init__(
        self,
        n_mix,
        layers=7,
        channels=32,
        use_log_freq=True,
        use_weighted_loss=True,
        use_binary_mask=True,
        reg_coef=0,
        reg_epsilon=None,
        reg2_coef=0,
        reg2_epsilon=None,
    ):
        assert n_mix == 2, "CLIPSepNIT supports only N = 2 for now."
        assert (
            use_binary_mask
        ), "CLIPSepNIT supports only use_binary_mask = True for now."
        if reg_coef > 0:
            assert (
                reg_epsilon is not None
            ), "Specify reg_epsilon when reg_coef > 0"
        if reg2_coef > 0:
            assert (
                reg2_epsilon is not None
            ), "Specify reg2_epsilon when reg2_coef > 0"

        super().__init__()
        self.n_mix = n_mix
        self.use_log_freq = use_log_freq
        self.use_weighted_loss = use_weighted_loss
        self.use_binary_mask = use_binary_mask
        self.reg_coef = reg_coef
        self.reg_epsilon = reg_epsilon
        self.reg2_coef = reg2_coef
        self.reg2_epsilon = reg2_epsilon

        # Create the neural net
        self.sound_net = UNet(in_dim=1, out_dim=channels, num_downs=layers)
        self.frame_net = nn.Linear(512, channels)
        self.pit_nets = torch.nn.ModuleList(
            [
                nn.Linear(channels, channels),
                nn.Linear(channels, channels),
            ]
        )
        self.synth_net = InnerProd(fc_dim=channels)

        # Initialize the weights
        self.sound_net.apply(init_weights)
        self.frame_net.apply(init_weights)
        self.pit_nets.apply(init_weights)
        self.synth_net.apply(init_weights)

    def forward(self, batch, img_emb, drop_closest=None, pit_loss=True):
        assert not drop_closest, "drop_closest not implemented!"

        N = self.n_mix
        mag_mix = batch["mag_mix"]
        mags = batch["mags"]

        # Pass through the frame net -> Bx1xC
        feat_frames_pre = [self.frame_net(img_emb[n]) for n in range(N)]
        feat_frames = [torch.sigmoid(feat) for feat in feat_frames_pre]

        mag_mix = mag_mix + 1e-10

        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # Warp the spectrogram
        if self.use_log_freq:
            grid_warp = torch.from_numpy(
                utils.warpgrid(B, 256, T, warp=True)
            ).to(mag_mix.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp, align_corners=True)

        # Calculate loss weighting coefficient (magnitude of input mixture)
        if self.use_weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # Compute ground truth masks after warping!
        gt_masks = [None] * N
        for n in range(N):
            if self.use_binary_mask:
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / sum(mags[n])
                gt_masks[n].clamp_(0.0, 1.0)

        # Compute log magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # Pass through the sound net -> BxCxHxW
        feat_sound = self.sound_net(log_mag_mix)

        # Pass through the synth net
        pred_masks = [
            self.synth_net(feat_frames[n], feat_sound) for n in range(N)
        ]

        # Get the input to the PIT stream
        mean_feat_frames_pre = sum(feat_frames_pre) / len(feat_frames_pre)
        feat_pit_pre = [net(mean_feat_frames_pre) for net in self.pit_nets]
        feat_pit = [torch.sigmoid(feat) for feat in feat_pit_pre]

        # Pass through the synth net for the PIT stream
        pit_masks = [self.synth_net(feat, feat_sound) for feat in feat_pit]

        # Mean activation
        mean_act = torch.mean(
            torch.sigmoid(pred_masks[0]) + torch.sigmoid(pred_masks[1])
        )
        mean_pit_act = torch.mean(
            torch.sigmoid(pit_masks[0]) + torch.sigmoid(pit_masks[1])
        )

        def _get_loss(x, n, gt):
            if n is None:
                pred = torch.sigmoid(x)
            elif x is None:
                pred = torch.sigmoid(n)
            else:
                pred = 1 - F.relu(1 - (torch.sigmoid(x) + torch.sigmoid(n)))
            return torch.mean(
                F.binary_cross_entropy(pred, gt, weight, reduction="none"),
                (1, 2, 3),
            )

        # Compute the loss
        if pit_loss:
            loss_00_0 = _get_loss(pred_masks[0], pit_masks[0], gt_masks[0])
            loss_11_1 = _get_loss(pred_masks[1], pit_masks[1], gt_masks[1])
            loss_0 = (loss_00_0 + loss_11_1) / 2
            loss_01_0 = _get_loss(pred_masks[0], pit_masks[1], gt_masks[0])
            loss_10_1 = _get_loss(pred_masks[1], pit_masks[0], gt_masks[1])
            loss_1 = (loss_01_0 + loss_10_1) / 2

            if self.reg_coef > 0:
                loss_x0_0 = _get_loss(None, pit_masks[0], gt_masks[0])
                loss_x1_1 = _get_loss(None, pit_masks[1], gt_masks[1])
                loss_0 += (
                    self.reg_coef
                    * (
                        F.relu(loss_00_0 - loss_x0_0 + self.reg_epsilon)
                        + F.relu(loss_11_1 - loss_x1_1 + self.reg_epsilon)
                    )
                    / 2
                )
                loss_x1_0 = _get_loss(None, pit_masks[1], gt_masks[0])
                loss_x0_1 = _get_loss(None, pit_masks[0], gt_masks[1])
                loss_1 += (
                    self.reg_coef
                    * (
                        F.relu(loss_01_0 - loss_x1_0 + self.reg_epsilon)
                        + F.relu(loss_10_1 - loss_x0_1 + self.reg_epsilon)
                    )
                    / 2
                )

            loss = torch.mean(torch.minimum(loss_0, loss_1))

            if self.reg2_coef > 0:
                loss += self.reg2_coef * F.relu(
                    mean_pit_act - self.reg2_epsilon, 0
                )

        else:
            loss = torch.mean(
                (
                    _get_loss(pred_masks[0], None, gt_masks[0])
                    + _get_loss(pred_masks[1], None, gt_masks[1])
                )
                / 2
            )

        return (
            loss,
            {
                "pred_masks": pred_masks,
                "pit_masks": pit_masks,
                "gt_masks": gt_masks,
                "mag_mix": mag_mix,
                "mags": mags,
                "weight": weight,
                "mean_act": mean_act,
                "mean_pit_act": mean_pit_act,
            },
        )

    def infer(self, mag_mix, img_emb, n_mix=1):
        N = n_mix

        # Pass through the frame net -> Bx1xC
        feat_frames_pre = [self.frame_net(img_emb[n]) for n in range(N)]
        feat_frames = [torch.sigmoid(feat) for feat in feat_frames_pre]

        mag_mix = mag_mix + 1e-10

        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # Warp the spectrogram
        if self.use_log_freq:
            grid_warp = torch.from_numpy(
                utils.warpgrid(B, 256, T, warp=True)
            ).to(mag_mix.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)

        # Compute log magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # Pass through the sound net -> BxCxHxW
        feat_sound = self.sound_net(log_mag_mix)

        # Pass through the synth net
        pred_masks = [
            self.synth_net(feat_frames[n], feat_sound) for n in range(N)
        ]

        # Get the input to the PIT stream
        mean_feat_frames_pre = sum(feat_frames_pre) / len(feat_frames_pre)
        feat_pit_pre = [net(mean_feat_frames_pre) for net in self.pit_nets]
        feat_pit = [torch.sigmoid(feat) for feat in feat_pit_pre]

        # Pass through the synth net for the PIT stream
        pit_masks = [self.synth_net(feat, feat_sound) for feat in feat_pit]

        return pred_masks, pit_masks

    def infer2(self, batch, img_emb):

        N = self.n_mix
        mag_mix = batch["mag_mix"]
        mags = batch["mags"]

        # Pass through the frame net -> Bx1xC
        feat_frames_pre = [self.frame_net(img_emb[0])]
        feat_frames = [torch.sigmoid(feat) for feat in feat_frames_pre]

        mag_mix = mag_mix + 1e-10

        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # Warp the spectrogram
        if self.use_log_freq:
            grid_warp = torch.from_numpy(
                utils.warpgrid(B, 256, T, warp=True)
            ).to(mag_mix.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp, align_corners=True)

        # Calculate loss weighting coefficient (magnitude of input mixture)
        if self.use_weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # Compute ground truth masks after warping!
        gt_masks = [None] * N
        for n in range(N):
            if self.use_binary_mask:
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / sum(mags[n])
                gt_masks[n].clamp_(0.0, 1.0)

        # Compute log magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # Pass through the sound net -> BxCxHxW
        feat_sound = self.sound_net(log_mag_mix)

        # Pass through the synth net
        pred_masks = [self.synth_net(feat_frames[0], feat_sound)]

        # Get the input to the PIT stream
        mean_feat_frames_pre = feat_frames_pre[0]
        feat_pit_pre = [net(mean_feat_frames_pre) for net in self.pit_nets]
        feat_pit = [torch.sigmoid(feat) for feat in feat_pit_pre]

        # Pass through the synth net for the PIT stream
        pit_masks = [self.synth_net(feat, feat_sound) for feat in feat_pit]

        # Mean activation
        mean_act = torch.mean(torch.sigmoid(pred_masks[0]))
        mean_pit_act = torch.mean(
            torch.sigmoid(pit_masks[0]) + torch.sigmoid(pit_masks[1])
        )

        return {
            "pred_masks": pred_masks,
            "pit_masks": pit_masks,
            "gt_masks": gt_masks,
            "mag_mix": mag_mix,
            "mags": mags,
            "weight": weight,
            "mean_act": mean_act,
            "mean_pit_act": mean_pit_act,
        }

    def infer3(self, batch, img_emb):

        mag_mix = batch["mag_mix"]

        # Pass through the frame net -> Bx1xC
        feat_frames_pre = [self.frame_net(img_emb)]
        feat_frames = [torch.sigmoid(feat) for feat in feat_frames_pre]

        mag_mix = mag_mix + 1e-10

        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # Warp the spectrogram
        if self.use_log_freq:
            grid_warp = torch.from_numpy(
                utils.warpgrid(B, 256, T, warp=True)
            ).to(mag_mix.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)

        # Calculate loss weighting coefficient (magnitude of input mixture)
        if self.use_weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # Compute log magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # Pass through the sound net -> BxCxHxW
        feat_sound = self.sound_net(log_mag_mix)

        # Pass through the synth net
        pred_masks = [self.synth_net(feat_frames[0], feat_sound)]

        # Get the input to the PIT stream
        mean_feat_frames_pre = feat_frames_pre[0]
        feat_pit_pre = [net(mean_feat_frames_pre) for net in self.pit_nets]
        feat_pit = [torch.sigmoid(feat) for feat in feat_pit_pre]

        # Pass through the synth net for the PIT stream
        pit_masks = [self.synth_net(feat, feat_sound) for feat in feat_pit]

        # Mean activation
        mean_act = torch.mean(torch.sigmoid(pred_masks[0]))
        mean_pit_act = torch.mean(
            torch.sigmoid(pit_masks[0]) + torch.sigmoid(pit_masks[1])
        )

        return {
            "pred_masks": pred_masks,
            "pit_masks": pit_masks,
            "mag_mix": mag_mix,
            "weight": weight,
            "mean_act": mean_act,
            "mean_pit_act": mean_pit_act,
        }

class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, pool_type="maxpool", dilate_scale=16):
        super().__init__()

        self.pool_type = pool_type

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                functools.partial(self._nostride_dilate, dilate=2)
            )
            orig_resnet.layer4.apply(
                functools.partial(self._nostride_dilate, dilate=4)
            )
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                functools.partial(self._nostride_dilate, dilate=2)
            )

        self.features = nn.Sequential(*list(orig_resnet.children())[:-2])

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            # Convolution layers with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # Other convolution layers
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, pool=True):
        x = self.features(x)

        if not pool:
            return x

        if self.pool_type == "avgpool":
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == "maxpool":
            x = F.adaptive_max_pool2d(x, 1)

        x = x.view(x.size(0), x.size(1))
        return x


class UNetBlock(nn.Module):
    """A U-Net block that defines the submodule with skip connection.

    X ---------------------identity-------------------- X
      |-- downsampling --| submodule |-- upsampling --|

    """

    def __init__(
        self,
        outer_nc,
        inner_input_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        use_dropout=False,
        inner_output_nc=None,
        noskip=False,
    ):
        super().__init__()
        self.outermost = outermost
        self.noskip = noskip
        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        if innermost:
            inner_output_nc = inner_input_nc
        elif inner_output_nc is None:
            inner_output_nc = 2 * inner_input_nc

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_input_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)
        upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        if outermost:
            downconv = nn.Conv2d(
                input_nc,
                inner_input_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3, padding=1
            )

            down = [downconv]
            up = [uprelu, upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            downconv = nn.Conv2d(
                input_nc,
                inner_input_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            upconv = nn.Conv2d(
                inner_output_nc,
                outer_nc,
                kernel_size=3,
                padding=1,
                bias=use_bias,
            )

            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            downconv = nn.Conv2d(
                input_nc,
                inner_input_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            upconv = nn.Conv2d(
                inner_output_nc,
                outer_nc,
                kernel_size=3,
                padding=1,
                bias=use_bias,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost or self.noskip:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UNet(nn.Module):
    """A UNet model."""

    def __init__(
        self,
        in_dim=1,
        out_dim=64,
        num_downs=5,
        ngf=64,
        use_dropout=False,
    ):
        super().__init__()

        # Construct the U-Net structure
        unet_block = UNetBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True
        )
        for i in range(num_downs - 5):
            unet_block = UNetBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                use_dropout=use_dropout,
            )
        unet_block = UNetBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block
        )
        unet_block = UNetBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block
        )
        unet_block = UNetBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block
        )
        unet_block = UNetBlock(
            out_dim,
            ngf,
            input_nc=in_dim,
            submodule=unet_block,
            outermost=True,
        )

        self.bn0 = nn.BatchNorm2d(in_dim)
        self.unet_block = unet_block

    def forward(self, x):
        x = self.bn0(x)
        x = self.unet_block(x)
        return x


class CondUNetBlock(nn.Module):
    """A U-Net block that defines the submodule with skip connection.

    X ---------------------identity-------------------- X
      |-- downsampling --| submodule |-- upsampling --|

    """

    def __init__(
        self,
        outer_nc,
        inner_input_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        inner_output_nc=None,
        noskip=False,
        cond_nc=None,
    ):
        super().__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.noskip = noskip
        self.cond_nc = cond_nc
        self.submodule = submodule

        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        if innermost:
            assert cond_nc > 0
            inner_output_nc = inner_input_nc + cond_nc
        elif inner_output_nc is None:
            inner_output_nc = 2 * inner_input_nc

        self.downnorm = nn.BatchNorm2d(inner_input_nc)
        self.uprelu = nn.ReLU(True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        if outermost:
            self.downconv = nn.Conv2d(
                input_nc,
                inner_input_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            self.upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3, padding=1
            )

        elif innermost:
            self.downrelu = nn.LeakyReLU(0.2, True)
            self.downconv = nn.Conv2d(
                input_nc,
                inner_input_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            self.upconv = nn.Conv2d(
                inner_output_nc,
                outer_nc,
                kernel_size=3,
                padding=1,
                bias=use_bias,
            )
            self.upnorm = nn.BatchNorm2d(outer_nc)

        else:
            self.downrelu = nn.LeakyReLU(0.2, True)
            self.downconv = nn.Conv2d(
                input_nc,
                inner_input_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            self.upconv = nn.Conv2d(
                inner_output_nc,
                outer_nc,
                kernel_size=3,
                padding=1,
                bias=use_bias,
            )
            self.upnorm = nn.BatchNorm2d(outer_nc)

    def forward(self, x, cond):
        if self.outermost:
            x_ = self.downconv(x)
            x_ = self.submodule(x_, cond)
            x_ = self.upconv(self.upsample(self.uprelu(x_)))

        elif self.innermost:
            x_ = self.downconv(self.downrelu(x))

            B, _, H, W = x_.size()
            cond_ = cond.unsqueeze(-1).unsqueeze(-1) * torch.ones(
                (B, self.cond_nc, H, W), device=x_.device
            )
            x_ = torch.concat((x_, cond_), 1)

            x_ = self.upnorm(self.upconv(self.upsample(self.uprelu(x_))))

        else:
            x_ = self.downnorm(self.downconv(self.downrelu(x)))
            x_ = self.submodule(x_, cond)
            x_ = self.upnorm(self.upconv(self.upsample(self.uprelu(x_))))

        if self.outermost or self.noskip:
            return x_
        else:
            return torch.cat([x, x_], 1)


class CondUNet(nn.Module):
    """A UNet model."""

    def __init__(
        self,
        in_dim=1,
        out_dim=64,
        cond_dim=32,
        num_downs=5,
        ngf=64,
        use_dropout=False,
    ):
        super().__init__()

        # Construct the U-Net structure
        unet_block = CondUNetBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            innermost=True,
            cond_nc=cond_dim,
        )
        for _ in range(num_downs - 5):
            unet_block = CondUNetBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block
            )
        unet_block = CondUNetBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block
        )
        unet_block = CondUNetBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block
        )
        unet_block = CondUNetBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block
        )
        unet_block = CondUNetBlock(
            out_dim,
            ngf,
            input_nc=in_dim,
            submodule=unet_block,
            outermost=True,
        )

        self.bn0 = nn.BatchNorm2d(in_dim)
        self.unet_block = unet_block

    def forward(self, x, cond):
        x = self.bn0(x)
        x = self.unet_block(x, cond)
        return x


class InnerProd(nn.Module):
    def __init__(self, fc_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(fc_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feat_img, feat_sound):
        sound_size = feat_sound.size()
        B, C = sound_size[0], sound_size[1]
        feat_img = feat_img.view(B, 1, C)
        z = torch.bmm(feat_img * self.scale, feat_sound.view(B, C, -1)).view(
            B, 1, *sound_size[2:]
        )
        z = z + self.bias
        return z

    def forward_nosum(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        feat_img = feat_img.view(B, C)
        z = (feat_img * self.scale).view(B, C, 1, 1) * feat_sound
        z = z + self.bias
        return z

    # inference purposes
    def forward_pixelwise(self, feats_img, feat_sound):
        (B, C, HI, WI) = feats_img.size()
        (B, C, HS, WS) = feat_sound.size()
        feats_img = feats_img.view(B, C, HI * WI)
        feats_img = feats_img.transpose(1, 2)
        feat_sound = feat_sound.view(B, C, HS * WS)
        z = torch.bmm(feats_img * self.scale, feat_sound).view(
            B, HI, WI, HS, WS
        )
        z = z + self.bias
        return z


class Bias(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        feat_img = feat_img.view(B, 1, C)
        z = torch.bmm(feat_img, feat_sound.view(B, C, H * W)).view(B, 1, H, W)
        z = z + self.bias
        return z

    def forward_nosum(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        z = feat_img.view(B, C, 1, 1) * feat_sound
        z = z + self.bias
        return z

    # inference purposes
    def forward_pixelwise(self, feats_img, feat_sound):
        (B, C, HI, WI) = feats_img.size()
        (B, C, HS, WS) = feat_sound.size()
        feats_img = feats_img.view(B, C, HI * WI)
        feats_img = feats_img.transpose(1, 2)
        feat_sound = feat_sound.view(B, C, HS * WS)
        z = torch.bmm(feats_img, feat_sound).view(B, HI, WI, HS, WS)
        z = z + self.bias
        return z


class FCPool(nn.Module):
    def __init__(self, fc_dim=64, pool_type="maxpool"):
        super().__init__()
        self.pool_type = pool_type
        self.fc = nn.Linear(512, fc_dim)

    def forward(self, x, pool=True):
        x = self.fc(x).permute(0, 2, 1)

        if not pool:
            return x

        if self.pool_type == "avgpool":
            x = F.adaptive_avg_pool1d(x, 1)
        elif self.pool_type == "maxpool":
            x = F.adaptive_max_pool1d(x, 1)

        x = x.view(x.size(0), x.size(1))

        return x
