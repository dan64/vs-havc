# Hybrid Automatic Video Colorizer (aka HAVC)

**Version 5.8.7**

A Deep Learning based [VapourSynth](https://www.vapoursynth.com/) filter for colorizing and restoring old images and video, based on the following projects: [DeOldify](https://github.com/jantic/DeOldify)
,  [DDColor](https://github.com/HolyWu/vs-ddcolor), [Colorization](https://github.com/richzhang/colorization), [Deep Exemplar based Video Colorization](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization), [DeepRemaster](https://github.com/satoshiiizuka/siggraphasia2019_remastering), [ColorMNet](https://github.com/yyang181/colormnet) and [CMNET2](https://github.com/dan64/cmnet2). The project  [Colorization](https://github.com/richzhang/colorization) includes 2 models: _Real-Time User-Guided Image Colorization with Learned Deep Priors_ (Zhang, 2017) and _Colorful Image Colorization_ (Zhang, 2016). These 2 models has been added as alternative models (named: _siggraph17_, _eccv16_) to DDColor.

The Vapoursynth filter version has the advantage of coloring the images directly in memory, without the need to use the filesystem to store the video frames.

For this filter is available a [User Guide](https://github.com/dan64/vs-havc/blob/main/documentation/HAVC%20User%20Guide.pdf) which provides useful tips and detailed explanations regarding the filter functions and usage. It is strongly recommended reading it before using the filter.

The filter (_HAVC_ in short) can be considered the _Swiss Army knife_ for coloring videos. It offers a wide range of options and coloring models and filters. It is able to combine the results provided by _DeOldify_ and _DDColor_ (_Colorization_), which are some of the best models available for coloring pictures, providing often a final colorized image that is better than the image obtained from the individual models. But the main strength of this filter is the addition of specialized filters to improve the quality of videos obtained by using these color models and the possibility to improve further the stability by using these models as input to [Deep Exemplar based Video Colorization](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization) model (_DeepEx_ in short), [DeepRemaster](https://github.com/satoshiiizuka/siggraphasia2019_remastering), [ColorMNet](https://github.com/yyang181/colormnet) and the new **CMNET2** model.

_DeepEx_, _DeepRemaster_, _ColorMNet_ and _[CMNET2](https://github.com/dan64/cmnet2)_ are exemplar-based video colorization models, which allow to colorize a movie starting from one or more external-colored reference images. They allow to colorize a Video in sequence based on the colorization history, enforcing its coherency by using a temporal consistency loss.

## What's New in 5.8.7 : CMNET2 Proximity Bias/DINOv3 backbone

The main addition of HAVC 5.8.7 are:

1) CMNET2 can now optionally favor temporally closer reference frames when several permanent-memory candidates match a frame's content similarly well (the same
situation described in [CMNET2 Memory Window](#cmnet2-memory-window) below), where a wide window holding visually similar but differently-colored references can wash
the result toward gray. Unlike the other CMNET2 options on this page, it is **not** a Hybrid field or a filter parameter: it is set once for the whole installation via
`models.json` (see [Model file names](#model-file-names-modelsjson) below), off by default. **DINOv3 only**, silently ignored for the legacy DINOv2 backbone. See
[CMNET2 Proximity Bias](#cmnet2-proximity-bias) below and the [full explanation and a visual example](https://github.com/dan64/cmnet2#proximity-weighted-memory-matching-optional-dinov3-only) in the CMNET2 README.

2) the support to the new **DINOv3 ViT-B/16** key-encoder backbone for all the CMNET2 based models: `HAVC_cmnet2()`, `HAVC_cmnet2dit()`, `HAVC_restore_video()` and `HAVC_deepex()` (with `ex_model=0`). The parameter is also exposed on the main entry point `HAVC_main()` as `DeepExBackbone` and propagated internally whenever CMNET2 is selected as the exemplar-based model (`DeepExModel=0`).

The new backbone is the **default** (`backbone="dinov3"`) in `HAVC_cmnet2()`, `HAVC_cmnet2dit()`, `HAVC_restore_video()` and `HAVC_deepex()`; the previous DINOv2 ViT-S/14 backbone is still available by setting `backbone="dinov2"`. On the main entry point `HAVC_main()`, for backward compatibility, `DeepExBackbone` still defaults to `"dinov2"`; set `DeepExBackbone="dinov3"` to opt into the new backbone.

The DINOv3 backbone is loaded by a native PyTorch implementation (no `transformers`, `huggingface_hub` or `tokenizers` dependency): the model is read from a local directory, self-contained like all the other weights of the project, never from the global HuggingFace cache.

Requirements: the new DINOv3 weights must be installed (see [Models Download](#models-download) below) together with the [safetensors](https://github.com/huggingface/safetensors) package (already included in the wheel dependencies).

## What's New in 5.8.5 : CMNET2-DiT

The main addition of HAVC 5.8.5 is `HAVC_cmnet2dit()`, a self-contained colorization filter that combines the **CMNET2** propagation model with a **DiT-based colorizer** (Nunchaku/Qwen-IE quantized, accessed via RPC). Unlike `HAVC_cmnet2()`, this function does not require a pre-colored reference clip : scene-change frames are extracted from the B&W input itself, colorized by the DiT engine in pairs, and loaded into the CMNET2 permanent-memory window as colored anchors.

For this filter has been developed in Hybrid a dedicated configuration page (named **CMNetDiT**) as shown in the following image:

![Model_cmnet2dit.jpg](https://github.com/dan64/vs-havc/blob/main/hybrid_setup/Model_cmnet2dit.jpg)

This means that it is possible colorize with a DiT model a B&W video with a single function call, with no external reference images needed.

Requirements: an external [DiTServerRPC](https://github.com/dan64/DiTServerRPC) server must be installed and running before the VapourSynth script is executed. A CUDA GPU with sufficient VRAM is required (**fp4** for RTX 50-Series, **int4** for RTX 30/40-Series).

See the dedicated section below and the [full documentation](documentation/HAVC_cmnet2dit.md) for details.

---

Additionally, a new `retry_model` parameter has been added to `HAVC_cmnet2()`,
`HAVC_deepex()`, and `HAVC_restore_video()`. It controls which colorization
engine is used by the CMNET2 retry path when a reference frame is missing:

| Value | Engine                            | Requires                               |
| ----- | --------------------------------- | -------------------------------------- |
| `0`   | HAVC (60% DeOldify + 40% DDColor) | Nothing extra (default)                |
| `1`   | DiT fp4                           | DiTServerRPC running, RTX 50-Series    |
| `2`   | DiT int4                          | DiTServerRPC running, RTX 30/40-Series |

If the selected DiT server is not reachable, the engine automatically falls back
to model `0` (HAVC).

## What's New in 5.8.0 : CMNET2

The major addition of HAVC 5.8.0 is **[CMNET2](https://github.com/dan64/cmnet2)**, a new exemplar-based video colorization model developed as an evolution of ColorMNet. CMNET2 is now the default exemplar model** in HAVC and provides significant improvements in color consistency and quality, especially on long videos with many reference frames. [CMNET2](https://github.com/dan64/cmnet2) was developed by me as an extension of ColorMNet in order to reduce the main defects of ColorNet: faded and/or incorrect colors for frames distant from the main reference frame (solved by the perm_mem), temporal inconsistency of colors (solved by the sliding perm_mem window). 

The key innovations of CMNET2 over ColorMNet are:

- **Three-tier memory architecture** (inspired by [XMem++](https://github.com/mbzuai-metaverse/XMem2)): a dedicated `perm_mem` store keeps the reference frames permanently, never compressing or evicting them. This ensures that the colors of the reference frames are preserved with high fidelity throughout the entire video, removing one of the main limitations of the original ColorMNet (which stored only key points of reference frames, allowing colors to drift away from the reference).
- **Sliding window over permanent memory**: long videos with hundreds or thousands of reference frames are handled by sliding the permanent-memory window forward as colorization progresses, keeping VRAM usage bounded without losing reference fidelity.
- **Adaptive VRAM management**: graduated response to memory pressure (slide 70% of permanent memory when VRAM is low, full reset only as a last resort), instead of the abrupt full memory reset of the original ColorMNet.
- **Reference preloading**: reference frames can be bulk-loaded into memory before colorization begins, decoupling the reference ingestion phase from the inference phase.
- 

Compared to ColorMNet, the practical effect is more **stable, consistent and faithful colors**, especially in long videos: the colors stay close to the reference frames over hundreds of frames instead of slowly drifting away, and new reference frames can be added during the video without losing the previous ones.

The standalone CMNET2 project (and full technical documentation) is available at: [github.com/dan64/cmnet2](https://github.com/dan64/cmnet2). The version integrated in HAVC 5.8.0 is exposed via the new public function `HAVC_cmnet2()` and is also used internally by `HAVC_main()` whenever the exemplar-based model is enabled.

> **NOTE for users coming from HAVC 5.6.7 or earlier**: the function `HAVC_cmnet2()` exists in earlier versions of HAVC, but in those versions it called the original ColorMNet model. **Starting with HAVC 5.8.0, `HAVC_cmnet2()` calls the new CMNET2 model.** Its parameter list has also changed (see the _Exemplar-based Models_ section). Existing scripts based on `HAVC_cmnet2()` should be reviewed before upgrading.

## Quick Start

To use the HAVC filter is necessary a GPU supporting CUDA, a NVIDIA RTX3060 is the minimum requirement to use this filter satisfactorily. The filter is distributed with the torch package provided with the **Hybrid Windows Addons**. To use it on Desktop (Windows) it is necessary install [Hybrid](https://www.selur.de/downloads) and the related [Addons](https://drive.google.com/drive/folders/1vC_pxwxL0o8fjmg8Okn0RA5rsodTcv9G?usp=drive_link). **Hybrid** is a Qt-based frontend for a lot video filters (including this one) which can convert most input formats to common audio & video formats and containers. Hybrid represents one of the most comprehensive solutions for implementing A.I. video filters and offers the most user-friendly approach to image colorization using the HAVC filter via [VapourSynth](https://www.vapoursynth.com/). In the folder _documentation_ is available a [User Guide](https://github.com/dan64/vs-havc/blob/main/documentation/HAVC%20User%20Guide.pdf) that provides detailed information on how to install Hybrid and use it to colorize videos. The Guide also provides tips on how to improve the final quality of colored movies.

## Dependencies

- [PyTorch](https://pytorch.org/get-started) 2.1.1 or later
- [VapourSynth](http://www.vapoursynth.com/) R62 or later
- [MiscFilters.dll](https://github.com/vapoursynth/vs-miscfilters-obsolete) Vapoursynth's Miscellaneous Filters
- [safetensors](https://github.com/huggingface/safetensors) 0.4 or later (used by the native DINOv3 backbone loader)

## Installation

```
pip install vshavc-x.x.x-py3-none-any.whl
```

with the version [5.8.0](https://github.com/dan64/vs-havc/releases/tag/v4.0.0) of HAVC has been released a modified version of DDColor to manage the Scene Detection properties available in the input clip, this version can be installed with the command:

```
pip install vsddcolor-1.0.2-py3-none-any.whl.zip
```

with the version 4.5.0 of HAVC has been introduced the support to ColorMNet. All the necessary packages to use ColorMNet (and CMNET2) are included in Hybrid's torch add-on package. For a manual installation not using Hybrid, it is necessary to install all the packages reported in the project page of [ColorMNet](https://github.com/yyang181/colormnet). To simplify the installation, in the release [4.5.0](https://github.com/dan64/vs-havc/releases/tag/v4.5.0) of this filter is available as asset the **spatial_correlation_sampler** package compiled against CUDA 12.4, python 3.12 and torch. To install it is necessary to unzip the following archive (using the nearest torch version available in the host system):

```
spatial_correlation_sampler-0.5.0-py312-cp312-win_amd64_torch-x.x.x.whl.zip
```

in the Library packages folder: .\Lib\site-packages\

## Models Download

The models are not installed with the package, they must be downloaded from the Deoldify website at: [completed-generator-weights](https://github.com/jantic/DeOldify#completed-generator-weights).

> The complete layout of all the weight files is summarized in [Weights folders (summary)](#weights-folders-summary) at the end of this section.

The models to download are:

- ColorizeVideo_gen.pth
- ColorizeStable_gen.pth
- ColorizeArtistic_gen.pth

The _model files_ have to be copied in the **models** directory usually located in:

.\Lib\site-packages\vshavc\models

To use ColorMNet and [CMNET2](https://github.com/dan64/cmnet2) it is also necessary to download the file [DINOv2FeatureV6_LocalAtten_s2_154000.pth](https://github.com/yyang181/colormnet/releases/download/v0.1/DINOv2FeatureV6_LocalAtten_s2_154000.pth) and save it in

.\Lib\site-packages\vshavc\colormnet\weights

A single copy is sufficient: [CMNET2](https://github.com/dan64/cmnet2) automatically locates and reuses the same file, so there is no need to duplicate it in the `colormnet2\weights` directory.

### DINOv3 backbone (default)

Starting from version 5.8.7 the CMNET2 based models use the **DINOv3 ViT-B/16** key-encoder backbone by default. Download these files from the [CMNET2 Releases](https://github.com/dan64/cmnet2/releases):

| File | Destination | Download |
|---|---|---|
| `DINOv3FeatureV6_LocalAtten_p374099.pth` | `.\Lib\site-packages\vshavc\colormnet2\weights\` | [download](https://github.com/dan64/cmnet2/releases/download/v1.3.0/DINOv3FeatureV6_LocalAtten_p374099.pth) |
| `dinov3-vitb16.zip` (extract to `vshavc\colormnet2\weights\`) | `.\Lib\site-packages\vshavc\colormnet2\weights\dinov3-vitb16\` | [download](https://github.com/dan64/cmnet2/releases/download/v1.1.0/dinov3-vitb16.zip) |

These files are used by CMNET2 only and are looked up in `vshavc\colormnet2\weights\`; the DINOv2 checkpoint of ColorMNet (CMNET1) stays in `vshavc\colormnet\weights\` and is reused by CMNET2 when `backbone="dinov2"` is selected (see [Model file names](#model-file-names-modelsjson) below).

> **Note:** the DINOv3 backbone is loaded by a native PyTorch implementation (`colormnet2/model/dinov3_vit.py`) from the local `dinov3-vitb16` directory: no `transformers` dependency and never from the global HuggingFace cache.

To keep using the previous **DINOv2 ViT-S/14** backbone, set `backbone="dinov2"` in the filter functions; the DINOv2 weights already installed for CMNET2 continue to be used.

### Model file names (`models.json`)

The names of the checkpoints are not hardcoded in the code: they are stored in a single data file, `vshavc\vsslib\models.json`, shipped with the package:

```json
{
  "cmnet2": {
    "dinov3": {
      "checkpoint": "DINOv3FeatureV6_LocalAtten_p374099.pth",
      "weights_root": "colormnet2",
      "weights_dir": "dinov3-vitb16",
      "enable_proximity_bias": false,
      "proximity_bias_alpha": 0.5
    },
    "dinov2": {
      "checkpoint": "DINOv2FeatureV6_LocalAtten_s2_154000.pth"
    }
  }
}
```

Normally there is no need to touch it. Edit it only if the checkpoint files have different names (custom or renamed weights), if the weights folders are different, or you want to change the default [proximity bias](#cmnet2-proximity-bias) settings: `checkpoint` is the file inside the weights directory of the backbone, `weights_root` is the package folder holding that weights directory (`colormnet2`: the DINOv3 files are CMNET2 only - it is missing in the `dinov2` entry because that checkpoint is shared with ColorMNet and stays in `vshavc\colormnet\weights\`), `weights_dir` is the auxiliary directory used by the DINOv3 backbone, and `enable_proximity_bias`/`proximity_bias_alpha` set the default for CMNET2's temporal-proximity-aware memory matching (DINOv3 only, ignored for DINOv2). When the configured file is missing, the filter stops immediately and the error lists the files actually present in the weights directory.

If `models.json` is missing or malformed, the built-in default names (the ones listed above) are used.

With the version 5.0 of HAVC has been added the model [DeepRemaster](https://github.com/satoshiiizuka/siggraphasia2019_remastering), for using it is necessary to download the file [remasternet.pth.tar](http://iizuka.cs.tsukuba.ac.jp/data/remasternet.pth.tar) (is not a tar, just a "pth" renamed as "pth.tar") and copy it in: ".\Lib\site-packages\vshavc\remaster\model".

At the first usage it is possible that are automatically downloaded by torch the neural networks: **resnet101** and **resnet34**, and starting with the release 4.5.0:  **resnet50**, **resnet18**, **dinov2_vits14_pretrain** and the folder **facebookresearch_dinov2_main**

So don't be worried if at the first usage the filter will be very slow to start, at the initialization are loaded almost all the _Fastai_ and _PyTorch_ modules and the resnet networks.

It is possible specify the destination directory of networks used by torch, by using the function parameter **torch\_dir**, if this parameter is set to **None**, the files will be downloaded in the _torch's cache_ dir, more details are available at: [caching-logic](https://pytorch.org/docs/stable/hub.html#caching-logic).

The models used by **DDColor** can be installed with the command

```
python -m vsddcolor
```

The models for **Deep-Exemplar based Video Colorization.** can be installed by downloading the file **colorization_checkpoint.zip** available in: [inference code](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization/releases/tag/v1.0).

The archive  **colorization_checkpoint.zip** have to be unziped in: .\Lib\site-packages\vshavc\deepex

### Weights folders (summary)

All the weight files present in an installation, in the directory tree of the `vshavc` package:

```text
.\Lib\site-packages\vshavc\
├── models\                             DeOldify + torch hub dir (default of "torch_dir")
│   ├── ColorizeVideo_gen.pth           DeOldify                  [manual download]
│   ├── ColorizeStable_gen.pth          DeOldify                  [manual download]
│   ├── ColorizeArtistic_gen.pth        DeOldify                  [manual download]
│   ├── colorization_release_v2-9b330a0b.pth   Zhang ECCV16       [automatic, torch]
│   ├── siggraph17-df00044c.pth                Zhang SIGGRAPH17   [automatic, torch]
│   ├── checkpoints\                    torch hub cache, filled at the first run   [automatic, torch]
│   │   ├── resnet18-5c106cde.pth       Fuse of ColorMNet/CMNET2
│   │   ├── resnet50-19c8e357.pth       Fuse of ColorMNet/CMNET2
│   │   ├── resnet101-63fe2227.pth      DeOldify (video, stable)
│   │   └── dinov2_vits14_pretrain.pth  legacy DINOv2 backbone
│   └── facebookresearch_dinov2_main\   legacy DINOv2 backbone (torch.hub repo)  [automatic, torch]
├── colormnet\weights\
│   └── DINOv2FeatureV6_LocalAtten_s2_154000.pth   ColorMNet (CMNET1), CMNET2 with backbone="dinov2"
├── colormnet2\weights\
│   ├── DINOv3FeatureV6_LocalAtten_p374099.pth     CMNET2 with backbone="dinov3" (default)
│   └── dinov3-vitb16\
│       ├── config.json
│       └── model.safetensors
├── deepex\                             DeepEx: unzip "colorization_checkpoint.zip" here
│   ├── checkpoints\
│   │   ├── video_moredata_l1\
│   │   │   ├── colornet_iter_76000.pth
│   │   │   ├── nonlocal_net_iter_76000.pth
│   │   │   └── discriminator_iter_76000.pth    [present, not loaded]
│   │   ├── resnet101-63fe2227.pth              [present, not loaded]
│   │   ├── resnet50-19c8e357.pth               [present, not loaded]
│   │   └── dinov2_vits14_pretrain.pth          [present, not loaded]
│   └── data\
│       ├── vgg19_conv.pth
│       └── vgg19_gray.pth              loaded at import time (always required)
└── remaster\model\
    └── remasternet.pth.tar             DeepRemaster               [manual download]
```

- **Manual download**: the links are in the sections above (DeOldify, ColorMNet/CMNET2 DINOv2, [DINOv3 backbone](#dinov3-backbone-default), DeepRemaster, Deep-Exemplar).
- **Automatic, torch**: downloaded by torch at the first run, inside the torch hub directory (the **torch\_dir** parameter, default `vshavc\models`); depending on the torch version the state dicts can be stored directly in `models\` or in its `checkpoints\` subfolder (in the tree above the two Zhang models are in `models\`, the resnets and DINOv2 in `checkpoints\`). With `torch_dir=None` the global _torch's cache_ dir is used instead; the `trusted_list` file that `torch.hub` creates in the same directory is not a weight.
- **Present, not loaded**: files that are part of the DeepEx folder but are never read by the filter: `discriminator_iter_76000.pth` (training discriminator) and the copies of the resnet/DINOv2 weights in `deepex\checkpoints\` - the filter loads those from `models\checkpoints\`. They can be left where they are.
- The DINOv2 checkpoint is shared: it is loaded by ColorMNet (CMNET1) and, from the same file, by CMNET2 when `backbone="dinov2"` is selected, even if CMNET2 lives in the `colormnet2` package folder (see [Model file names](#model-file-names-modelsjson)).
- The models of **DDColor** are not stored here: DDColor is the separate `vsddcolor` package (installed as a dependency of HAVC) and its models are downloaded with the command `python -m vsddcolor`.

## Usage

```python
# loading plugins
import vshavc as havc

# changing range from limited to full range for HAVC
clip = core.resize.Bicubic(clip, range_in_s="limited", range_s="full")
# setting color range to PC (full) range.
clip = core.std.SetFrameProps(clip=clip, _ColorRange=0)
# adjusting color space from YUV420P16 to RGB24
clip = core.resize.Bicubic(clip=clip, format=vs.RGB24, matrix_in_s="709", range_s="full")


# DeOldify with DDColor, Preset = "fast"
clip = havc.HAVC_main(clip=clip, Preset="fast")
# DeOldify only model
clip = havc.HAVC_colorizer(clip, method=0)
# DDColor only model
clip = havc.HAVC_colorizer(clip, method=1)

# To apply video color stabilization filters to colored clip
clip = havc.HAVC_stabilizer(clip, dark=True, smooth=True, stab=True)

# Simplest way to use Presets
clip = havc.HAVC_main(clip=clip, Preset="fast", ColorFix="violet/red", ColorTune="medium", ColorMap="none")

# CMNET2 model (default exemplar model in 5.8.0) using HAVC as input for the reference frames
clip = havc.HAVC_main(clip=clip, EnableDeepEx=True, ScThreshold=0.1)

# changing range from full to limited range for HAVC
clip = core.resize.Bicubic(clip, range_in_s="full", range_s="limited")
```

See `__init__.py` for the description of the parameters.

**NOTES**:

- In the _DDColor_ version included with **HAVC** the parameter _input_size_ has changed name in _render_factor_ because were changed the range of values of this parameter to be equivalent to _render_factor_ in _DeOldify_, the relationship between these 2 parameters is the following:

```
input_size = render_factor * 16
```

- In the modified version of _DDColor_ 1.0.1 was added the boolean parameter _scenechange_, if this parameter is set to _True_, will be colored only the frames tagged as scene change.

- In the folder [samples](https://github.com/dan64/vs-havc/tree/main/samples) there are some clips and reference images that can be used to test the filter. The clips _sample_colored_sync.mp4_ and _sample_colored_async.mp4_ are useful to test the new video restore functionality added in HAVC 5.0 (described in the User Guide). The clip _sample_colored_sync.mp4_ is fully in sync with the clip _sample_bw_.mp4 and any of the exemplar-based models can be used to colorize it, while the clip _sample_colored_async.mp4_ is not in sync and only _DeepRemaster_ is able to properly colorize the movie.

## Filter Usage

The filter was developed having in mind to use it mainly to colorize movies. Both DeOldify and DDcolor are good models for coloring pictures (see the _Comparison of Models_). But when are used for coloring movies they are introducing artifacts that usually are not noticeable in the images. Especially in dark scenes both DeOldify and DDcolor are not able to understand what it is the dark area and what color to give it, they often decide to color these dark areas with blue, then in the next frame this area could become red and then in the next frame return to blue, introducing a flashing psychedelic effect when all the frames are put in a movie.
To try to solve this problem has been developed _pre-_ and _post-_ process filters. It is possible to see them in the Hybrid screenshot below.

![Hybrid Coloring page](https://github.com/dan64/vs-havc/blob/main/hybrid_setup/Model_D%2BD_filters.JPG)

The main filters introduced are:

**Chroma Smoothing**: This filter allows to reduce the _vibrancy_ of colors assigned by DeOldify/DDcolor by using the parameters _de-saturation_ and _de-vibrancy_ (the effect on _vibrancy_ will be visible only if the option **chroma resize** is enabled, otherwise this parameter has effect on the _luminosity_). The area impacted by the filter is defined by the thresholds dark/white. All the pixels with luma below the dark threshold will be impacted by the filter, while the pixels above the white threshold will be left untouched. All the pixels in the middle will be gradually impacted depending on the luma value.

**Chroma Stabilization**: This filter will try to stabilize the frames' colors. As explained previously since the frames are colored individually, the colors can change significantly from one frame to the next, introducing a disturbing psychedelic flashing effect. This filter try to reduce this by averaging the chroma component of the frames. The average is performed using a number of frames specified in the _Frames_ parameter.
Are implemented 2 averaging methods:

1. _Arithmetic average_: the current frame is averaged using equal weights on the past and future frames
2. _Weighted average_: the current frame is averaged using a weighed mean of the past and future frames, where the weight decrease with the time (far frames have lower weight respect to the nearest frames).

As explained previously the stabilization is performed by averaging the past/future frames. Since the non matched areas of past/future frames are _gray_ because is missing in the past/future the _color information_, the filter will apply a _color restore_ procedure that fills the gray areas with the pixels of current frames (eventually de-saturated with the parameter "sat"). The image restored in this way is blended with the non restored image using the parameter "weight". The gray areas are selected by the threshold parameter "tht". All the pixels in the HSV color space with "S" < "tht" will be considered gray. If is detected a scene change (controlled by the parameter "tht_scen"), the _color restore_ is not applied.

**DDColor Tweaks**: This filter is available only for DDColor and has been added because has been observed that the DDcolor's _inference_ is quite poor on dark/bright scenes depending on the luma value. This filter will force the luma of input image to don't be below the threshold defined by the parameter _luma_min_. Moreover this filter allows to apply a dynamic gamma correction. The gamma adjustment will be applied when the average luma is below the parameter _gamma_luma_min_. A _gamma_ value > 2.0 improves the DDColor stability on bright scenes, while a _gamma_ < 1 improves the DDColor stability on dark scenes.

**B&W tune**: Starting with HAVC version 5.5.0, a new post-processing filter called B&W Tune was introduced, which can automatically correct most color allocation errors. The color adjustment capability has been further improved with the version 5.6.0 where the Retinex filter and LUTs have been included to improve the overall output color quality. Unfortunately, forcing color stability has the side effect of producing washed-out colors with a slight pink cast (similar to skin tone). This new post-processing filter can automatically correct this problem and restore image colors to a more natural color. With the version 5.6.0, HAVC has evolved beyond basic colorization, it now delivers vivid, natural colors while ensuring consistent color stability throughout films.

### Chroma Adjustment

Unfortunately when are applied to movies the color models are subject to assign unstable colors to the frames especially on the red/violet chroma range. This problem is more visible on DDColor than on DeOldify.
To mitigate this issue was necessary to implement some kind of chroma adjustment. This adjustment allows to de-saturate all the colors included in a given color range. The color range must be specified in the HSV color space. This color space is useful because all the chroma is represented by only the parameter "Hue". In this color space the colors are specified in degree (from 0 to 360), as shown in the [DDeoldify Hue Wheel](https://github.com/dan64/vs-havc/blob/main/hybrid_setup/ddeoldify_hue_wheel.jpg).
It is possible to apply this adjustment on all filters described previously.
Depending on the filter the adjustment can be enabled using the following syntax:

```
chroma_range = "hue_start:hue_end" or "hue_wheel_name"
```

for example this assignment:

```
chroma_range = "290:330,rose"
```

specify the range of hue colors: 290-360, because "rose" is [hue wheel name](https://github.com/dan64/vs-havc/blob/main/hybrid_setup/ddeoldify_hue_wheel.jpg) that correspond to the range:330-360.

It is possible to specify more ranges by using the comma "," separator.

When the de-saturation information is not already available in the filter's parameters, it necessary to use the following syntax:

```
chroma_adjustment = "chroma_range|sat,weight"
```

in this case it is necessary to specify also the de-saturation parameter "sat" and the blending parameter "weight".

for example with this assignment:

```
chroma_range = "300:340|0.4,0.2"
```

the hue colors in the range 300-340 will be de-saturated by the amount 0.4 and the final frame will be blended by applying a 20% de-saturation of 0.4 an all the pixels (if weight=0, no blending is applied).

To simplify the usage of this filter has been added the Preset _ColorFix_ which allows to fix a given range of chroma combination. The strength of the filter is controlled by the the Preset _ColorTune_.

#### Color Mapping

Using an approach similar to _Chroma Adjustment_ has been introduced the possibility to remap a given gange of colors in another chroma range. This remapping is controlled by the Preset _ColorMap_. For example the preset "blue->brown" allows to remap all the chroma combinations of _blue_ in the color _brown_. It is not expected that this filter can be applied on a full movie, but it could be useful to remap the color on some portion of a movie.

In the [HAVC User Guide](https://github.com/dan64/vs-havc/blob/main/documentation/HAVC%20User%20Guide.pdf) are provided useful tips on how to use both the _Chroma Adjustment_ and _Color Mapping_ features provided by this filter.

### Merging the models

As explained previously, this filter is able to combine the results provided by DeOldify and DDColor, to perform this combination has been implemented 8 methods:

0. _DeOldify_ only coloring model.

1. _DDColor_ only color model.

2. _Simple Merge_: the frames are combined using a _weighted merge_, where the parameter _merge_weight_ represent the weight assigned to the frames provided by the DDcolor model.

3. _Constrained Chroma Merge_: given that the colors provided by DeOldify's _Video_ model are more conservative and stable than the colors obtained with DDcolor. The frames are combined by assigning a limit to the amount of difference in chroma values between DeOldify and DDcolor. This limit is defined by the parameter _threshold_. The limit is applied to the frame converted to "YUV". For example when threshold=0.1, the chroma values "U","V" of DDcolor frame will be constrained to have an absolute percentage difference respect to "U","V" provided by DeOldify not higher than 10%. If _merge_weight_ is < 1.0, the chroma limited DDColor frames will be will be merged again with the frames of DeOldify using the _Simple Merge_.
   
   4. _Luma Masked Merge_: the behaviour is similar to the method _Adaptive Luma Merge_. With this method the frames are combined using a _masked merge_. The pixels of DDColor's frame with luma < _luma_limit_ will be filled with the (de-saturated) pixels of DeOldify, while the pixels above the _white_limit_ threshold will be left untouched. All the pixels in the middle will be gradually replaced depending on the luma value. If the parameter  _merge_weight_ is < 1.0, the resulting masked frames will be merged again with the non de-saturated frames of DeOldify using the _Simple Merge_.

4. _Adaptive Luma Merge_: given that the DDcolor performance is quite bad on dark scenes, with this method the images are combined by decreasing the weight assigned to DDcolor frames when the luma is below the _luma_threshold_. For example with: luma_threshold = 0.6 and alpha = 1, the weight assigned to DDcolor frames will start to decrease linearly when the luma < 60% till _min_weight_. For _alpha_=2, the weight begins to decrease quadratically.

5. _Chroma Retention Merge_: Given that the colors provided by deoldify() are more conservative and stable than the colors obtained with ddcolor(). This function try to restore the colors of gray pixels provide by deoldify() by using the colors provided by ddcolor(). The gray pixels are identified by the parameter "tht". Once are identified the gray pixels are substituted with the desaturated colors in deoldify(), the level of desaturation is identified by the parameter "sat". It is performed a "gradient" substitution, i.e. the gray pixels are gradually substituted depending on the level of gray gradient. The steepness of gradient curve is controlled by the parameter "alpha". Optionally is possible to resize the frame before the filter application to speed up the filter by setting True the parameter ?chroma_resize?.

6. _ChromaBound Adaptive_: Adaptive version of Constrained-Chroma. In this version the chroma tolerance is adaptive, i.e., it is applied an approach that will allow more color variation in textured/complex regions and less in smooth areas. The texture strength is computed via Laplacian and chroma tolerance is controlled by the following parameters:
   
   - base_tol: int = 20,  Base chroma tolerance (smooth areas)
   - max_extra: int = 24,  Extra tolerance for textured areas

The merging methods 2-7 are leveraging on the fact that usually the DeOldify _Video_ model provides frames which are more stable, this feature is exploited to stabilize also DDColor. The methods 3, 4 and 7 are similar to _Simple Merge_, but before the merge with _DeOldify_ the _DDColor_ frame is limited in the chroma changes (method 3) or limited based on the luma (method 4). The method 5 is a _Simple Merge_ where the weight decrease with luma and the method 7 is an hybrid model that combines the approach of methods 4 and 5.

## Comparison of Models

Taking inspiration from the article published on Habr: [Mode on: Comparing the two best colorization AI's](https://habr.com/en/companies/ruvds/articles/568426/). It was decided to use it to get the refence images and the images obtained using the [ColTran](https://github.com/google-research/google-research/tree/master/coltran) model, to extend the analysis with the models implemented in the **HAVC** filter.

The added models are:

**D+D**: DeOldify (with model _Video_ & render_factor = 24) + DDColor (with model _Artistic_ and render_factor = 24)
![Hybrid D+D](https://github.com/dan64/vs-havc/blob/main/hybrid_setup/Model_D%2BD.JPG)

**DD**: DDColor (with model _Artistic_ and and render_factor = 24 equivalent to input_size = 384)
![Hybrid_DD](https://github.com/dan64/vs-havc/blob/main/hybrid_setup/Model_DD.JPG)

**DS**: DeOldify (with model _Stable_ & render_factor =24)
![Hybrid D+D](https://github.com/dan64/vs-havc/blob/main/hybrid_setup/Model_DS.JPG)

**DV**: DeOldify (with model _Video_ & render_factor = 24)
![Hybrid D+D](https://github.com/dan64/vs-havc/blob/main/hybrid_setup/Model_DV.JPG)

**T241**: ColTran + TensorFlow 2.4.1 model as shown in [Habr](https://habr.com/en/companies/ruvds/articles/568426/)

### Summary of Results

The models were compared by measuring the perceptual color distance (**dE**) of each colorized image from a ground-truth reference using the [CIEDE2000](https://en.wikipedia.org/wiki/Color_difference#CIEDE2000) method, which takes into account the non-uniformities of human color perception.

The combined model **D+D** (DeOldify + DDColor) was the best overall performer, winning 10 out of 23 tests in the first set and confirming that DeOldify and DDColor are able to compensate each other's weaknesses. The **DD** model (DDColor alone) was the second best, but with occasional poor outputs that the merge with DeOldify is able to correct. **T241** (ColTran) was the worst performer.

In a second test set focused on combinations of DeOldify _Artistic_ and _Stable_ with different DDColor variants, all the combined models performed similarly well, confirming the positive impact of merging the two model families regardless of the specific variants used.

The full per-image **CIEDE2000** results, the comparison methodology and the detailed analysis of both test sets are available in [documentation/MODEL_COMPARISON.md](https://github.com/dan64/vs-havc/blob/main/documentation/MODEL_COMPARISON.md).

## Exemplar-based Models

As stated previously to stabilize further the colorized videos it is possible to use the frames colored by HAVC as reference frames (exemplar) as input to the supported exemplar-based models: [CMNET2](https://github.com/dan64/cmnet2), [ColorMNet](https://github.com/yyang181/colormnet), [Deep Exemplar based Video Colorization](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization) and [DeepRemaster](https://github.com/satoshiiizuka/siggraphasia2019_remastering).

In Hybrid the _Exemplar Models_ have their own panel, as shown in the following picture:
![Hybrid DeepEx](https://github.com/dan64/vs-havc/blob/main/hybrid_setup/Model_DeepEx.JPG)

The available exemplar models are selected via the field **Model** with the following values:

- 0 : **CMNET2** (default) : new in 5.8.0, recommended
- 1 : **Deep-Exemplar**
- 2 : **DeepRemaster**
- 3 : **ColorMNet** (original)

For CMNET2 and ColorMNet there are 2 implementations defined, by the field **Mode**:

- 'remote' (has not memory frames limitation but it uses a remote process for the inference)
- 'local' (the inference is performed inside the Vapoursynth local thread but has memory limitation)

The field **Preset** controls the render method and speed, allowed values are:

- 'Auto' (default : automatically assigns the optimal render size)
- 'Fast' (faster but colors are more washed out)
- 'Medium' (colors are a little washed out)
- 'Slow' (slower but colors are a little more vivid)
- 'Slower' (colors are more accurate, usually very slow)

The 'Auto' and 'Slower' presets are new in 5.8.0.

The field **SC thresh** define the sensitivity for the scene detection (suggested value **0.1**, see [Miscellaneous Filters](https://amusementclub.github.io/doc3/plugins/misc.html)), while the field **SC min freq** allows to specify the minimum number of reference frames that have to be generated.

The flag **Vivid** has different meanings depending on the _Exemplar Model_ used:

- **CMNET2**: the saturation will be increased by about 15%.
- **Deep-Exemplar**: the saturation will be increased by about 25%.
- **DeepRemaster**: the saturation will be increased by about 20% and Hue by +10.
- **ColorMNet**: the frames memory is reset at every reference frame update.

The field **Method** allows to specify the type of reference frames (RF) provided in input to the _Exemplar-based Models_, allowed values are:

- 0 = HAVC same as video (default)
- 1 = HAVC + RF same as video
- 2 = HAVC + RF different from video
- 3 = external RF same as video
- 4 = external RF different from video
- 5 = external ClipRef same as video
- 6 = external ClipRef different from video

It is possible to specify the directory containing the external reference frames by using the field **Ref FrameDir**. The frames must be named using the following format: _ref_nnnnnn.[png|jpg]_. For the methods 5 and 6 it is possible to pass a video clip as source for reference images.

Unfortunately the exemplar-based methods other than [CMNET2](https://github.com/dan64/cmnet2) have the problem that they are unable to properly colorize the new "features" (new elements not available in the reference frame) so that often these new elements are colored with implausible colors (see for an example: [New "features" are not properly colored](https://github.com/yyang181/NTIRE23-VIDEO-COLORIZATION/issues/10)). To try to fix this problem has been introduced the possibility to merge the frames propagated by the exemplar model with the frames colored with DDColor and/or DeOldify. The merge is controlled by the field **Ref merge**, allowed values are:

- 0 = no merge
- 1 = reference frames are merged with low weight
- 2 = reference frames are merged with medium weight
- 3 = reference frames are merged with high weight

When the field **Ref merge** is set to a value greater than 0, the field **SC min freq** is set =1, to allows the merge for every frame (more details are provided in [HAVC User Guide](https://github.com/dan64/vs-havc/blob/main/documentation/HAVC%20User%20Guide.pdf)). Note that **Ref merge** is intended for the exemplar models that suffer from the "new features" problem; with [CMNET2](https://github.com/dan64/cmnet2) it is not disabled but is not useful, so the recommended value is 0.

Finally the flag **Reference frames only** can be used to export the reference frames generated with the method **HAVC** and defined by the parameters **SC thresh**, **SC min freq** fields.

### CMNET2 Memory Window

When [CMNET2](https://github.com/dan64/cmnet2) is used as exemplar model, the parameter **DeepExMaxMemFrames** controls the size of the **sliding permanent-memory window**: it defines how many reference frames are held in the model's permanent memory at any given time. As colorization progresses, the window slides forward automatically, evicting the oldest references and loading new ones. This allows CMNET2 to handle long videos with hundreds of reference frames while keeping VRAM usage bounded.

Suggested values for CMNET2:

- min = 10, max = 500
- if = 0 (default), the window size is automatically set to 50

For ColorMNet and DeepRemaster, **DeepExMaxMemFrames** keeps its previous meaning (max number of encoded frames / max number of reference frames in memory). Note that the suggested ranges have been revised in 5.8.0; please refer to the docstring of `HAVC_main()` and to the [User Guide](https://github.com/dan64/vs-havc/blob/main/documentation/HAVC%20User%20Guide.pdf) for the up-to-date values.

### CMNET2 Retry Model

When `retry_threshold > 0`, CMNET2 may request a fresh colorized reference frame
for frames with insufficient permanent-memory coverage. The `retry_model`
parameter selects the engine used for this on-the-fly colorization:

| Value | Engine                    | Requires                               |
| ----- | ------------------------- | -------------------------------------- |
| `0`   | HAVC (DeOldify + DDColor) | Nothing extra (default)                |
| `1`   | DiT fp4                   | DiTServerRPC running, RTX 50-Series    |
| `2`   | DiT int4                  | DiTServerRPC running, RTX 30/40-Series |

If the selected DiT server is not reachable, the engine automatically falls back
to model `0` (HAVC). See [HAVC_cmnet2dit.md](documentation/HAVC_cmnet2dit.md) for
details on DiT setup.

### CMNET2 Proximity Bias

By default, CMNET2 ranks permanent-memory candidates purely by content similarity, with no notion of *when* in the video a reference frame was captured relative to the
frame being colorized with a wide memory window (see [CMNET2 Memory Window](#cmnet2-memory-window) above) holding several visually similar
but differently-colored references, this can wash the result toward gray. CMNET2 can optionally favor temporally closer references instead, without
ever reducing the permanent memory’s overall contribution to the readout. Unlike the other options on this page, this is **not** a Hybrid field or a function
parameter of `HAVC_main()`/`HAVC_cmnet2()`/`HAVC_cmnet2dit()` - it is configured once for the whole installation through the `enable_proximity_bias`/`proximity_bias_alpha`
keys in `models.json` (see [Model file names](#model-file-names-modelsjson) above), off by default. 
To permanently enable it (useful for permanent memory window size > 50) it is necessary to set `enable_proximity_bias=true` in the configuration 
file stored in: `vsslib/models.json` as shown in the example below:

```json
{
  "cmnet2": {
    "dinov3": {
      "checkpoint": "DINOv3FeatureV6_LocalAtten_p374099.pth",
      "weights_root": "colormnet2",
      "weights_dir": "dinov3-vitb16",
      "enable_proximity_bias": true,
      "proximity_bias_alpha": 0.5
    },
    "dinov2": {
      "checkpoint": "DINOv2FeatureV6_LocalAtten_s2_154000.pth"
    }
  }
}
```

**DINOv3 only**: the setting is silently ignored when `backbone="dinov2"`.
See the [mechanism explanation and a visual example](https://github.com/dan64/cmnet2#proximity-weighted-memory-matching-optional-dinov3-only) in the CMNET2 README.

## HAVC_cmnet2dit : Self-contained DiT + CMNET2 Colorization

> **Full documentation:** [documentation/HAVC_cmnet2dit.md](documentation/HAVC_cmnet2dit.md) -
> detailed guide with advanced usage examples, model configuration, and
> troubleshooting tips.

`HAVC_cmnet2dit()` integrates a DiT-based colorization model with CMNET2,
making the colorization pipeline self-contained : no pre-colored reference
clip is needed.

### Architecture

```
B&W input clip
     |
     +-- SceneDetectEdges() --> B&W scene-change frames (ref frames)
     |                                    |
     |                     HAVCditEngine.colorize_image_pair()
     |                                    |
     |                     Colored ref frames -> CMNET2 perm_mem window
     |                                    |
     +------------> CMNET2 propagation ----> Colored output clip
```

Reference frames are colorized **in pairs** by the DiT model (Nunchaku-qwen
with fp4/int4 quantization), halving the number of DiT forward passes. A
`colorize_image()` fallback handles any odd leftover.

### Requirements

- **[DiTServerRPC](https://github.com/dan64/DiTServerRPC)** : external RPC
  server that loads the DiT model and exposes the colorization API. Must be
  installed and running before the VapourSynth script.
- CUDA GPU with sufficient VRAM (RTX 50-Series: fp4, RTX 30/40-Series: int4).
- CMNET2 model weights accessible at the package model directory (DINOv3 backbone by default, see [Models Download](#models-download)).

### Basic Usage

```python
import vshavc as havc

clip_bw = havc.HAVC_read_video(source="video_bw.mp4")
clip = havc.HAVC_cmnet2dit(clip_bw)
```

### Key Parameters

| Parameter           | Default     | Description                                                      |
| ------------------- | ----------- | ---------------------------------------------------------------- |
| `render_speed`      | `auto`      | CMNET2 render resolution: Auto / Fast / Medium / Slow / Slower   |
| `render_vivid`      | `False`     | +15% saturation boost after colorization                         |
| `sc_thresh`         | `0.035`     | Scene edges-detection threshold [0.01, 0.15]                     |
| `sc_tht_ssim`       | `0.80`      | SSIM threshold to filter similar scene-change frames             |
| `sc_min_int`        | `25`        | Minimum frame distance between scene changes                     |
| `sc_tht_offset`     | `2`         | Frame offset for scene-change comparison [1-25]                  |
| `sc_min_freq`       | `0`         | Force at least 1 ref frame every N frames [0-1000]               |
| `encode_mode`       | `0`         | 0=remote CMNET2 backend (recommended), 1=local                   |
| `max_memory_frames` | `20`        | Sliding permanent-memory window size (rounded to even)           |
| `dit_engine_params` | `None`      | DiT model configuration dict (see below)                         |
| `torch_dir`         | `model_dir` | Torch hub dir for CMNET2 model weights                           |
| `retry_threshold`   | `0.0`       | Threshold for retry with additional ref frames [0-1], 0=disabled |
| `retry_model`       | `1`         | Model for retry: 0=HAVC, 1=DiT fp4, 2=DiT int4                   |

### `dit_engine_params` Keys

| Key                     | Default                    | Description                                     |
| ----------------------- | -------------------------- | ----------------------------------------------- |
| `host`                  | `127.0.0.1`                | DiTServerRPC address                            |
| `port`                  | `8765`                     | DiTServerRPC port                               |
| `model_inference_steps` | `4`                        | Steps used to select the model file to download |
| `cache_dir`             | `""`                       | HuggingFace cache directory                     |
| `model_name`            | `nunchaku-qwen`            | Nunchaku model name                             |
| `model_precision`       | `fp4`                      | `fp4` (RTX 50xx) or `int4` (RTX 30/40xx)        |
| `model_rank`            | `32`                       | SVD rank: `32` or `128`                         |
| `full_model_path`       | `""`                       | Absolute path to a local .safetensors file      |
| `prompt`                | `"Colorize this image..."` | Text prompt guiding colorization style          |
| `steps`                 | `2`                        | DiT inference steps per image                   |
| `img_size`              | `0`                        | Max long-side in pixels (0=original size)       |

> **Note:** `retry_model=1` (default for this function) selects DiT fp4, matching
> `model_precision="fp4"` in `dit_engine_params`. Use `retry_model=2` for int4
> or `retry_model=0` for HAVC fallback.
> 
> If `retry_model` and `dit_engine_params["model_precision"]` are in conflict
> (e.g. `retry_model=1` but `model_precision="int4"`), the value in
> `dit_engine_params` takes priority : the DiT server singleton is initialized
> from the dict first, and `retry_model` cannot override an already-loaded model.

### Performance

Typical ~4 fps on RTX 50-Series with default settings. When
`host="127.0.0.1"`, image transfer uses shared memory (zero-copy, ~23%
faster than remote hosts). `encode_mode=0` (remote CMNET2) is recommended
to avoid GPU memory contention between DiT and CMNET2.

### Differences from `HAVC_cmnet2()`

| Feature                            | `HAVC_cmnet2`                            | `HAVC_cmnet2dit`                      |
| ---------------------------------- | ---------------------------------------- | ------------------------------------- |
| Reference frames                   | Pre-colored (caller provides `clip_ref`) | B&W, derived from input clip          |
| Reference colorization             | n/a                                      | `HAVCditEngine.colorize_image_pair()` |
| `clip_ref` parameter               | Required                                 | Not exposed                           |
| `sc_framedir`                      | Supported                                | Not supported                         |
| `colormap`/`dark`/`smooth` filters | Supported                                | Not supported                         |
| `ref_merge`                        | Supported                                | Not supported                         |
| Window size                        | Any positive integer                     | Always even (pair-wise scheme)        |

## Coloring using Hybrid

As stated previously the simplest way to colorize images with the HAVC filter it to use [Hybrid](https://www.selur.de/downloads). To simplify the usage has been introduced standard Presets that automatically apply all the filter's settings. A set of parameters that are able to provide a satisfactory colorization are the following:

- **Speed:** slower
- **Color map:** red->brown
- **Color tweaks:** retinex/red
- **Denoise:** medium
- **Stabilize:** balanced
- **B&W tune:** light
- **B&W mode:** CLAHE (luma)
- **Interpolation:** 3

then enable the _Exemplar Models_ check box and set

- **Method:** HAVC
- **Model:** CMNET2
- **SC thresh:** 0.10
- **SC SSIM thresh:** 0.0
- **SC min freq:** 0
- **normalize:** checked
- **Mode:** remote
- **Frames:** 0
- **Preset:** Auto
- **Vivid:** checked
- **Ref merge:** high

In the following picture are shown the suggested parameters:

![Hybrid Preset](https://github.com/dan64/vs-havc/blob/main/hybrid_setup/Model_Presets.JPG)

The suggested settings are appropriate for a medium powered GPU (RTX4070 or above). In the [HAVC User Guide](https://github.com/dan64/vs-havc/blob/main/documentation/HAVC%20User%20Guide.pdf) are provided more settings depending on the available hardware.

## Conclusions

In Summary **HAVC** is able to provide often a final colorized image that is better than the image obtained from the individual models, and can be considered an improvement respect to the current Models. With the introduction of **[CMNET2](https://github.com/dan64/cmnet2)** in version 5.8.0, the temporal consistency and color fidelity over long videos have been further improved, making HAVC an even more solid choice for video colorization. It is highly recommended to read the [HAVC User Guide](https://github.com/dan64/vs-havc/blob/main/documentation/HAVC%20User%20Guide.pdf) which provides useful tips on how to improve the colored movies.

As a final consideration I would like to point out that the test results showed that the images coloring technology is mature enough to be used concretely both for coloring images and, thanks to **Hybrid**, videos.

## Acknowledgements

I would like to thank Selur, author of [Hybrid](https://www.selur.de/), for his wise advices and for having developed a gorgeous interface for this filter. Despite the large number of parameters and the complexity of managing them appropriately, the interface developed by Selur makes its use easy even for non-experts users.
