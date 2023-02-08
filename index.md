<center>
<a href="https://salu133445.github.io/">Hao-Wen Dong</a>†,
 <a href="https://kotonaoya.wixsite.com/naoya-takahashi/">Naoya Takahashi</a>*,
 <a href="https://www.yukimitsufuji.com/">Yuki Mitsufuji</a>,
 <a href="https://cseweb.ucsd.edu/~jmcauley/"> Julian McAuley</a>,
 <a href="https://cseweb.ucsd.edu/~tberg/"> Taylor Berg-Kirkpatrick</a><BR>
In International Conference on Learning Representations (ICLR) 2023<BR>
 <span style="font-size: 80%;">(† Work done during an internship at Sony Group Corporation, * corresponding author)</span><br>
<a href="https://arxiv.org/abs/2212.07065">paper</a> | <a href="https://github.com/sony/CLIPSep">code</a>|
</center>

## Content

- [Example results on “MUSIC + VGGSound”](#music-vggsound)
  1. ["accordion" + "engine accelerating"](#music-vggsound-1)
  2. ["acoustic guitar" + "cheetah chirrup"](#music-vggsound-2)
  3. ["violin" + "people sobbing"](#music-vggsound-3)
- [Example results on "VGGSound-Clean + VGGSound”](#vggsound-vggsound)
  1. ["cat growling" + "railroad car"](#vggsound-vggsound-1)
  2. ["electric grinder" + "car horn"](#vggsound-vggsound-2)
  3. ["playing harpsichord" + "people coughing"](#vggsound-vggsound-2)
- [Example results on “VGGSound + None”](#vggsound)
  1. ["playing bagpipes"](#vggsound-1)
  2. ["subway, metro, underground"](#vggsound-2)
  3. ["playing theremin"](#vggsound-3)
- [Real-world movie example](#movie)
  1. [Spiderman -- No Way Home (2021)](#movie-1)
- [Robustness to different queries](#queries)
  1. ["acoustic guitar" + "cheetah chirrup"](#queries-1)

---

## Summary of the compared models

- __CLIPSep__: Our proposed model _without_ the noise invariant training.
- __CLIPSep-NIT__: Our proposed model with the noise invariant training using γ = 0.25.
- __LabelSep__: The proposed CLIPSep model with the query model replaced by a learnable embedding lookup table.
- __PIT__: The permutation invariant training model proposed by Yu et al. (2017).[^yu2017]

| Model | Unlabelled data | Post-processing free | Query type (training) | Query type (test) |
|-|:-:|:-:|:-:|:-:|
| CLIPSep | ✓ | ✓ | Image | Text |
| CLIPSep-NIT | ✓ | ✓ | Image | Text |
| LabelSep | ✕ | ✓ | Label | Label |
| PIT | ✓ | ✕ | - | - |
{:style="width: 75%; margin-left: auto; margin-right: auto;"}

[^yu2017]: Dong Yu, Morten Kolbæk, Zheng-Hua Tan, and Jesper Jensen. Permutation invariant training of deep models for speaker-independent multi-talker speech separation. In _Proc. ICASSP_, 2017.

---

## Important notes

- All the examples presented below use _text queries_{:.red} rather than image queries.
- We prefix the text query into the form of "_a photo of [user input query]_".
- All the spectrograms are shown in the log frequency scale.

---

## Example results on "MUSIC + VGGSound" {#music-vggsound}

> __Settings__: We take an audio sample in the MUSIC dataset as the _target source_. We then mix the target source with an _interference_ audio sample in the VGGSound dataset to create an artificial mixture.

### Example 1 -- "accordion" + "engine accelerating" {#music-vggsound-1}

- __Target source__: accordion
- __Interference__: engine accelerating revving vroom
- __Query__: "_accordion_{:.red}"

| Mixture | Ground truth | Ground truth (Interference) | Prediction (CLIPSep) |
|:-:|:-:|:-:|:-:|
| ![mix.png](music-vggsound/accordion-PohzBumrY2Q+K-8KURljE8_FA/mix.png){:.spec} | ![gtamp.png](music-vggsound/accordion-PohzBumrY2Q+K-8KURljE8_FA/gtamp.png){:.spec} | ![intamp.png](music-vggsound/accordion-PohzBumrY2Q+K-8KURljE8_FA/intamp.png){:.spec} | ![predmap.png](music-vggsound/accordion-PohzBumrY2Q+K-8KURljE8_FA/clipsep/predamp.png){:.spec} |
| {% include audio_player.html filename="music-vggsound/accordion-PohzBumrY2Q+K-8KURljE8_FA/mix.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/accordion-PohzBumrY2Q+K-8KURljE8_FA/gt.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/accordion-PohzBumrY2Q+K-8KURljE8_FA/int.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/accordion-PohzBumrY2Q+K-8KURljE8_FA/clipsep/pred.wav" style="max-width: 240px;" %} |

| Prediction (CLIPSep-NIT) | Prediction (PIT) | Noise head 1 (CLIPSep-NIT) _\*_{:.red} | Noise head 2 (CLIPSep-NIT) _\*_{:.red} |
|:-:|:-:|:-:|:-:|
| ![predamp.png](music-vggsound/accordion-PohzBumrY2Q+K-8KURljE8_FA/predamp.png){:.spec} | ![predamp.png](music-vggsound/accordion-PohzBumrY2Q+K-8KURljE8_FA/pit/predamp.png){:.spec} | ![pitmag1.png](music-vggsound/accordion-PohzBumrY2Q+K-8KURljE8_FA/pitmag1.png){:.spec} | ![pitmag2.png](music-vggsound/accordion-PohzBumrY2Q+K-8KURljE8_FA/pitmag2.png){:.spec} |
| {% include audio_player.html filename="music-vggsound/accordion-PohzBumrY2Q+K-8KURljE8_FA/pred.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/accordion-PohzBumrY2Q+K-8KURljE8_FA/pit/pred.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/accordion-PohzBumrY2Q+K-8KURljE8_FA/pit0.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/accordion-PohzBumrY2Q+K-8KURljE8_FA/pit1.wav" style="max-width: 240px;" %} |

> _\*_{:.red} The noise heads are expected to contain query-irrelevant noises.

### Example 2 -- "acoustic guitar" + "cheetah chirrup" {#music-vggsound-2}

- __Target source__: acoustic guitar
- __Interference__: cheetah chirrup
- __Query__: "_acoustic guitar_{:.red}"

| Mixture | Ground truth | Ground truth (Interference) | Prediction (CLIPSep) |
|:-:|:-:|:-:|:-:|
| ![mix.png](music-vggsound/acoustic_guitar-X4UWEifacwI+H-2HSizQ0OJHM/mix.png){:.spec} | ![gtamp.png](music-vggsound/acoustic_guitar-X4UWEifacwI+H-2HSizQ0OJHM/gtamp.png){:.spec} | ![intamp.png](music-vggsound/acoustic_guitar-X4UWEifacwI+H-2HSizQ0OJHM/intamp.png){:.spec} | ![predmap.png](music-vggsound/acoustic_guitar-X4UWEifacwI+H-2HSizQ0OJHM/clipsep/predamp.png){:.spec} |
| {% include audio_player.html filename="music-vggsound/acoustic_guitar-X4UWEifacwI+H-2HSizQ0OJHM/mix.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/acoustic_guitar-X4UWEifacwI+H-2HSizQ0OJHM/gt.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/acoustic_guitar-X4UWEifacwI+H-2HSizQ0OJHM/int.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/acoustic_guitar-X4UWEifacwI+H-2HSizQ0OJHM/clipsep/pred.wav" style="max-width: 240px;" %} |

| Prediction (CLIPSep-NIT) | Prediction (PIT) _\*_{:.red} | Noise head 1 (CLIPSep-NIT) | Noise head 2 (CLIPSep-NIT) |
|:-:|:-:|:-:|:-:|
| ![predamp.png](music-vggsound/acoustic_guitar-X4UWEifacwI+H-2HSizQ0OJHM/predamp.png){:.spec} | ![predamp.png](music-vggsound/acoustic_guitar-X4UWEifacwI+H-2HSizQ0OJHM/pit/predamp.png){:.spec} | ![pitmag1.png](music-vggsound/acoustic_guitar-X4UWEifacwI+H-2HSizQ0OJHM/pitmag1.png){:.spec} | ![pitmag2.png](music-vggsound/acoustic_guitar-X4UWEifacwI+H-2HSizQ0OJHM/pitmag2.png){:.spec} |
| {% include audio_player.html filename="music-vggsound/acoustic_guitar-X4UWEifacwI+H-2HSizQ0OJHM/pred.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/acoustic_guitar-X4UWEifacwI+H-2HSizQ0OJHM/pit/pred.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/acoustic_guitar-X4UWEifacwI+H-2HSizQ0OJHM/pit0.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/acoustic_guitar-X4UWEifacwI+H-2HSizQ0OJHM/pit1.wav" style="max-width: 240px;" %} |

> _\*_{:.red} The PIT model requires a post-selection step to get the correct source. Without the post-selection step, the PIT model return the right source in only a 50% chance.

### Example 3 -- "violin" + "people sobbing" {#music-vggsound-3}

- __Target source__: violin
- __Interference__: people sobbing
- __Query__: "_violin_{:.red}"

| Mixture | Ground truth | Ground truth (Interference) | Prediction (CLIPSep) |
|:-:|:-:|:-:|:-:|
| ![mix.png](music-vggsound/violin-WjwOm5pJlI0+3-O3HG59u1yek/mix.png){:.spec} | ![gtamp.png](music-vggsound/violin-WjwOm5pJlI0+3-O3HG59u1yek/gtamp.png){:.spec} | ![intamp.png](music-vggsound/violin-WjwOm5pJlI0+3-O3HG59u1yek/intamp.png){:.spec} | ![predmap.png](music-vggsound/violin-WjwOm5pJlI0+3-O3HG59u1yek/clipsep/predamp.png){:.spec} |
| {% include audio_player.html filename="music-vggsound/violin-WjwOm5pJlI0+3-O3HG59u1yek/mix.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/violin-WjwOm5pJlI0+3-O3HG59u1yek/gt.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/violin-WjwOm5pJlI0+3-O3HG59u1yek/int.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/violin-WjwOm5pJlI0+3-O3HG59u1yek/clipsep/pred.wav" style="max-width: 240px;" %} |

| Prediction (CLIPSep-NIT) | Prediction (PIT) | Noise head 1 (CLIPSep-NIT) | Noise head 2 (CLIPSep-NIT) |
|:-:|:-:|:-:|:-:|
| ![predamp.png](music-vggsound/violin-WjwOm5pJlI0+3-O3HG59u1yek/predamp.png){:.spec} | ![predamp.png](music-vggsound/violin-WjwOm5pJlI0+3-O3HG59u1yek/pit/predamp.png){:.spec} | ![pitmag1.png](music-vggsound/violin-WjwOm5pJlI0+3-O3HG59u1yek/pitmag1.png){:.spec} | ![pitmag2.png](music-vggsound/violin-WjwOm5pJlI0+3-O3HG59u1yek/pitmag2.png){:.spec} |
| {% include audio_player.html filename="music-vggsound/violin-WjwOm5pJlI0+3-O3HG59u1yek/pred.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/violin-WjwOm5pJlI0+3-O3HG59u1yek/pit/pred.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/violin-WjwOm5pJlI0+3-O3HG59u1yek/pit0.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="music-vggsound/violin-WjwOm5pJlI0+3-O3HG59u1yek/pit1.wav" style="max-width: 240px;" %} |

---

## Example results on "VGGSound-Clean + VGGSound" {#vggsound-vggsound}

> __Settings__: We take an audio sample in the VGGSound-Clean dataset as the _target source_. We then mix the target source with an _interference_ audio sample in the VGGSound dataset to create an artificial mixture. Note that the LabelSep model does not work on the MUSIC dataset due to the different label taxonomies of the MUSIC and VGGSound datasets.

### Example 1 -- "cat growling" + "railroad car" {#vggsound-vggsound-1}

- __Target source__: cat growling
- __Interference__: railroad car train wagon
- __Query__: "_cat growling_{:.red}"

| Mixture | Ground truth | Ground truth (Interference) |
|:-:|:-:|:-:|
| ![mix.png](vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/mix.png){:.spec} | ![gtamp.png](vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/gtamp.png){:.spec} | ![intamp.png](vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/intamp.png){:.spec} |
| {% include audio_player.html filename="vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/mix.wav" %} | {% include audio_player.html filename="vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/gt.wav" %} | {% include audio_player.html filename="vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/int.wav" %}

| Prediction (CLIPSep) | Prediction (CLIPSep-NIT) | Prediction (PIT) |
|:-:|:-:|:-:|
| ![predmap.png](vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/clipsep/predamp.png){:.spec} | ![predamp.png](vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/predamp.png){:.spec} | ![predamp.png](vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/pit/predamp.png){:.spec} |
| {% include audio_player.html filename="vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/clipsep/pred.wav" %} | {% include audio_player.html filename="vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/pred.wav" %} | {% include audio_player.html filename="vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/pit/pred.wav" %} |

| Prediction (LabelSep) | Noise head 1 (CLIPSep-NIT) _\*_{:.red} | Noise head 2 (CLIPSep-NIT) _\*_{:.red} |
|:-:|:-:|:-:|
| ![predamp.png](vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/labelsep/predamp.png){:.spec} | ![pitmag1.png](vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/pitmag1.png){:.spec} | ![pitmag2.png](vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/pitmag2.png){:.spec} |
| {% include audio_player.html filename="vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/labelsep/pred.wav" %} | {% include audio_player.html filename="vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/pit0.wav" %} | {% include audio_player.html filename="vggsound-vggsound/G-gGeyOIugeEs+M-lMei2nMceX4/pit1.wav" %} |

> _\*_{:.red} The noise heads are expected to contain query-irrelevant noises.

### Example 2 -- "electric grinder" + "car horn" {#vggsound-vggsound-2}

- __Target source__: electric grinder grinding
- __Interference__: vehicle horn car horn honking
- __Query__: "_electric grinder grinding_{:.red}"

| Mixture | Ground truth | Ground truth (Interference)
|:-:|:-:|:-:|
| ![mix.png](vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/mix.png){:.spec} | ![gtamp.png](vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/gtamp.png){:.spec} | ![intamp.png](vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/intamp.png){:.spec} |
| {% include audio_player.html filename="vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/mix.wav" %} | {% include audio_player.html filename="vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/gt.wav" %} | {% include audio_player.html filename="vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/int.wav" %}

| Prediction (CLIPSep) | Prediction (CLIPSep-NIT) | Prediction (PIT) _\*_{:.red} |
|:-:|:-:|:-:|
| ![predmap.png](vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/clipsep/predamp.png){:.spec} | ![predamp.png](vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/predamp.png){:.spec} | ![predamp.png](vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/pit/predamp.png){:.spec} |
| {% include audio_player.html filename="vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/clipsep/pred.wav" %} | {% include audio_player.html filename="vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/pred.wav" %} | {% include audio_player.html filename="vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/pit/pred.wav" %} |

| Prediction (LabelSep) | Noise head 1 (CLIPSep-NIT) | Noise head 2 (CLIPSep-NIT) |
|:-:|:-:|:-:|
| ![predamp.png](vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/labelsep/predamp.png){:.spec} | ![pitmag1.png](vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/pitmag1.png){:.spec} | ![pitmag2.png](vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/pitmag2.png){:.spec} |
| {% include audio_player.html filename="vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/labelsep/pred.wav" %} | {% include audio_player.html filename="vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/pit0.wav" %} | {% include audio_player.html filename="vggsound-vggsound/u-6uCeOOJXK24+X-lX8Csm309Ls/pit1.wav" %} |

> _\*_{:.red} The PIT model requires a post-selection step to get the correct source. Without the post-selection step, the PIT model return the right source in only a 50% chance.

### Example 3 -- "playing harpsichord" + "people coughing" {#vggsound-vggsound-3}

- __Target source__: playing harpsichord
- __Interference__: people coughing
- __Query__: "_playing harpsichord_{:.red}"

| Mixture | Ground truth | Ground truth (Interference)
|:-:|:-:|:-:|
| ![mix.png](vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/mix.png){:.spec} | ![gtamp.png](vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/gtamp.png){:.spec} | ![intamp.png](vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/intamp.png){:.spec} |
| {% include audio_player.html filename="vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/mix.wav" %} | {% include audio_player.html filename="vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/gt.wav" %} | {% include audio_player.html filename="vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/int.wav" %}

| Prediction (CLIPSep) | Prediction (CLIPSep-NIT) | Prediction (PIT) _\*_{:.red} |
|:-:|:-:|:-:|
| ![predmap.png](vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/clipsep/predamp.png){:.spec} | ![predamp.png](vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/predamp.png){:.spec} | ![predamp.png](vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/pit/predamp.png){:.spec} |
| {% include audio_player.html filename="vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/clipsep/pred.wav" %} | {% include audio_player.html filename="vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/pred.wav" %} | {% include audio_player.html filename="vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/pit/pred.wav" %} |

| Prediction (LabelSep) | Noise head 1 (CLIPSep-NIT) | Noise head 2 (CLIPSep-NIT) |
|:-:|:-:|:-:|
| ![predamp.png](vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/labelsep/predamp.png){:.spec} | ![pitmag1.png](vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/pitmag1.png){:.spec} | ![pitmag2.png](vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/pitmag2.png){:.spec} |
| {% include audio_player.html filename="vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/labelsep/pred.wav" %} | {% include audio_player.html filename="vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/pit0.wav" %} | {% include audio_player.html filename="vggsound-vggsound/3-x3HIdaOkLQ4+3-43FtU9nHMxc/pit1.wav" %} |

> _\*_{:.red} The PIT model requires a post-selection step to get the correct source. Without the post-selection step, the PIT model return the right source in only a 50% chance.

---

## Example results on "VGGSound + None" {#vggsound}

> __Settings__: We take a "noisy" audio sample in the VGGSound dataset and treat it as the input mixture. We aim to examine if the model can separate the target sounds from query-irrelevant noises. Note that there is no "ground truth" in this setting.

### Example 1 -- "playing bagpipes" {#vggsound-1}

- __Target source__: playing bagpipes ([Bagpipe player in Vegas
](https://www.youtube.com/watch?v=hvCj8Dk0Su4&t=5))
- __Interference__: none
- __Query__: "_playing bagpipes_{:.red}"
- __Note__: The model successfully separates the bagpipe sounds and the background noises.

| Source video |
|:-:|
| {% include video_player.html filename="vggsound/v-hvCj8Dk0Su4/av.mp4" width=200 height=200 %} |

| Mixture | Prediction | Noise head 1 | Noise head 2 |
|:-:|:-:|:-:|:-:|
| ![mix.png](vggsound/v-hvCj8Dk0Su4/mix.png){:.spec} | ![predamp.png](vggsound/v-hvCj8Dk0Su4/predamp.png){:.spec} | ![pitmag1.png](vggsound/v-hvCj8Dk0Su4/pitmag1.png){:.spec} | ![pitmag2.png](vggsound/v-hvCj8Dk0Su4/pitmag2.png){:.spec} |
| {% include audio_player.html filename="vggsound/v-hvCj8Dk0Su4/mix.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="vggsound/v-hvCj8Dk0Su4/pred.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="vggsound/v-hvCj8Dk0Su4/pit0.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="vggsound/v-hvCj8Dk0Su4/pit1.wav" style="max-width: 240px;" %} |

### Example 2 -- "subway, metro, underground" {#vggsound-2}

- __Target source__: subway, metro, underground ([Trains At Flitwick Railway Station (03/5/15)](https://www.youtube.com/watch?v=xe7opvK-LJg&t=373s))
- __Interference__: none
- __Query__: "_subway, metro, underground_{:.red}"
- __Note__: The model successfully separates the wind sounds from the train sounds.

| Source video |
|:-:|
| {% include video_player.html filename="vggsound/e-xe7opvK-LJg/av.mp4" width=200 height=200 %} |

| Mixture | Prediction | Noise head 1 | Noise head 2 |
|:-:|:-:|:-:|:-:|
| ![mix.png](vggsound/e-xe7opvK-LJg/mix.png){:.spec} | ![predamp.png](vggsound/e-xe7opvK-LJg/predamp.png){:.spec} | ![pitmag1.png](vggsound/e-xe7opvK-LJg/pitmag1.png){:.spec} | ![pitmag2.png](vggsound/e-xe7opvK-LJg/pitmag2.png){:.spec} |
| {% include audio_player.html filename="vggsound/e-xe7opvK-LJg/mix.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="vggsound/e-xe7opvK-LJg/pred.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="vggsound/e-xe7opvK-LJg/pit0.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="vggsound/e-xe7opvK-LJg/pit1.wav" style="max-width: 240px;" %} |

### Example 3 -- "playing theremin" {#vggsound-3}

- __Target source__: playing theremin ([Терменвокс](https://www.youtube.com/watch?v=23kti2LQUdw&t=71s))
- __Interference__: none
- __Query__: "_playing theremin_{:.red}"
- __Note__: The model successfully separates most theremin sounds from the piano accompaniments.

| Source video |
|:-:|
| {% include video_player.html filename="vggsound/3-23kti2LQUdw/av.mp4" width=200 height=200 %} |

| Mixture | Prediction | Noise head 1 | Noise head 2 |
|:-:|:-:|:-:|:-:|
| ![mix.png](vggsound/3-23kti2LQUdw/mix.png){:.spec} | ![predamp.png](vggsound/3-23kti2LQUdw/predamp.png){:.spec} | ![pitmag1.png](vggsound/3-23kti2LQUdw/pitmag1.png){:.spec} | ![pitmag2.png](vggsound/3-23kti2LQUdw/pitmag2.png){:.spec} |
| {% include audio_player.html filename="vggsound/3-23kti2LQUdw/mix.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="vggsound/3-23kti2LQUdw/pred.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="vggsound/3-23kti2LQUdw/pit0.wav" style="max-width: 240px;" %} | {% include audio_player.html filename="vggsound/3-23kti2LQUdw/pit1.wav" style="max-width: 240px;" %} |

---

## Real-world movie example {#movie}

> __Settings__: We take the audio track of a commercial movie clip as the input mixture. We aim to examine if the model can extract the desired sounds corresponding to a specific query. We use the CLIPSep model without the noise invariant training in this demo. Note that we use _text queries_{:.red} rather than image queries.

### "Spiderman -- No Way Home (2021)" {#movie-1}

- __Source__: [Spider-Man: No Way Home (2021) - Curing the Villains Scene (9/10) - Movieclips](https://www.youtube.com/watch?v=A4kZ2Nnsm_g)
- __Note__: While the model is trained on the VGGSound dataset, the model generalizes to commercial movie soundtracks despite the large domain gap.

| Source video |
|:-:|
| {% include video_player.html filename="spiderman-no-way-home/spiderman-no-way-home.mp4" width=720 %} |

| Prediction<br>(Query: "_orchestra_{:.red}") |
|:-:|
| {% include video_player.html filename="spiderman-no-way-home/orchestra.mp4" width=720 %} |

| Prediction<br>(Query: "_people yelling_{:.red}") |
|:-:|
| {% include video_player.html filename="spiderman-no-way-home/people_yelling.mp4" width=720 %} |

| Prediction<br>(Query: "_orchestra_{:.red} and _people yelling_{:.red}") |
|:-:|
| {% include video_player.html filename="spiderman-no-way-home/orchestra_people_yelling.mp4" width=720 %} |
| (In the final example, we see how we can _combine __multiple queries__ to extract multiple target sounds_.) |

---

## Robustness to different queries {#queries}

> __Settings__: We take the same input mixture and query the model with different _text queries_{:.red} to examine the model's robustness to different queries. We use the CLIPSep-NIT model in this demo.

### "acoustic guitar" + "cheetah chirrup" {#queries-1}

- __Target source__: acoustic guitar
- __Interference__: cheetah chirrup
- __Note__: We can see that the model is robust to different text queries and can extract the desired sounds.

| Mixture | Ground truth | Ground truth (Interference) |
|:-:|:-:|:-:|
| ![mix.png](queries/audio-X4UWEifacwI+audio-2HSizQ0OJHM0/mix.png){:.spec} | ![gtamp.png](queries/audio-X4UWEifacwI+audio-2HSizQ0OJHM0/gtamp.png){:.spec} | ![intamp.png](queries/audio-X4UWEifacwI+audio-2HSizQ0OJHM0/intamp.png){:.spec} |
| {% include audio_player.html filename="queries/audio-X4UWEifacwI+audio-2HSizQ0OJHM0/mix.wav" %} | {% include audio_player.html filename="queries/audio-X4UWEifacwI+audio-2HSizQ0OJHM0/gt.wav" %} | {% include audio_player.html filename="queries/audio-X4UWEifacwI+audio-2HSizQ0OJHM0/int.wav" %} |

| Prediction<br>(Query: "_acoustic guitar_{:.red}") | Prediction<br>(Query: "_guitar_{:.red}") | Prediction<br>(Query: "_a man is playing acoustic guitar_{:.red}") |
|:-:|:-:|:-:|
| ![predmap.png](queries/audio-X4UWEifacwI+audio-2HSizQ0OJHM0/predamp.png){:.spec} | ![predmap.png](queries/audio-X4UWEifacwI+audio-2HSizQ0OJHM1/predamp.png){:.spec} | ![predmap.png](queries/audio-X4UWEifacwI+audio-2HSizQ0OJHM2/predamp.png){:.spec} |
| {% include audio_player.html filename="queries/audio-X4UWEifacwI+audio-2HSizQ0OJHM0/pred.wav" %} | {% include audio_player.html filename="queries/audio-X4UWEifacwI+audio-2HSizQ0OJHM1/pred.wav" %} | {% include audio_player.html filename="queries/audio-X4UWEifacwI+audio-2HSizQ0OJHM2/pred.wav" %} |

| Prediction<br>(Query: "_a man is playing acoustic guitar in a room_{:.red}") | Prediction<br>(Query: "_car engine_{:.red}") |
|:-:|:-:|
| ![predmap.png](queries/audio-X4UWEifacwI+audio-2HSizQ0OJHM3/predamp.png){:.spec} | ![predmap.png](queries/audio-X4UWEifacwI+audio-2HSizQ0OJHM4/predamp.png){:.spec} |
| {% include audio_player.html filename="queries/audio-X4UWEifacwI+audio-2HSizQ0OJHM3/pred.wav" %} | {% include audio_player.html filename="queries/audio-X4UWEifacwI+audio-2HSizQ0OJHM4/pred.wav" %} |

---
