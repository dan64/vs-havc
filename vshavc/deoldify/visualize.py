"""
------------------------------------------------------------------------------- 
Author: Jason Antic
Date: 2019-08-20
version: 
LastEditors: Dan64
LastEditTime: 2025-11-24
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
module that contains the main functions to control deoldify's colorization process.
This module has been modified to be compatible with Vapoursynth.
All the interface functions has been changed to work with PIL images.
A new class ModelImageRender has been added to manage the colorization.
The new ImageRender will try to adapt the models "Stable" and "Artistic" to
video colorization by blending the colored frame with the model "Video".
"""
from ..fastai.core import *
from ..fastai.vision import *
from .filters import IFilter, MasterFilter, ColorizerFilter
from .generators import gen_inference_deep, gen_inference_wide
from PIL import Image
import requests

class ModelImageVisualizer:

    def __init__(self, filter: IFilter):
        self.filter = filter

    def _clean_mem(self):
        torch.cuda.empty_cache()
        # gc.collect()

    def get_transformed_image(self, orig_image: Image, render_factor: int = None, post_process: bool = True) -> Image:
                
        self._clean_mem()
        filtered_image = self.filter.filter(orig_image, orig_image, render_factor=render_factor, post_process=post_process)

        return filtered_image

class ModelImageRender:
    _model_video = None
    _model_stable = None
    _model_artistic = None
    _video_weight = 0
    _render_factor_video: int = 0
    _render_factor_stable: int = 0
    _render_factor_artistic: int = 0
    _modelname : str = ''

    package_dir='./'

    def __init__(self, package_dir: str = None, modelname: str = 'video', render_factor: int = 24,
                 video_weight: float = 0):

        self.package_dir = package_dir

        self._modelname = modelname

        self._video_weight = video_weight

        if modelname == 'video':
            self.init_video_colorizer(render_factor=render_factor)
        elif modelname == 'stable':
            self.init_video_colorizer(render_factor=render_factor)
            self.init_stable_colorizer(render_factor=render_factor)
        else:
            self.init_video_colorizer(render_factor=render_factor)
            self.init_artistic_colorizer(render_factor=render_factor)

    def init_video_colorizer(self, render_factor: int = 24):

        root_folder = Path(self.package_dir)

        weights_name: str = 'ColorizeVideo_gen'

        if self._model_video is not None and render_factor == self._render_factor_video:
            return   # nothing to do

        learn = gen_inference_wide(root_folder=root_folder, weights_name=weights_name)
        filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)

        self._model_video = ModelImageVisualizer(filtr)
        self._render_factor_video = render_factor


    def init_stable_colorizer(self,  render_factor: int = 24):

        root_folder = Path(self.package_dir)

        weights_name: str = 'ColorizeStable_gen'

        if self._model_stable is not None and render_factor==self._render_factor_stable:
            return   # nothing to do

        learn = gen_inference_wide(root_folder=root_folder, weights_name=weights_name)
        filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)

        self._model_stable = ModelImageVisualizer(filtr)
        self._render_factor_stable = render_factor


    def init_artistic_colorizer(self, render_factor: int = 24) :

        root_folder = Path(self.package_dir)

        weights_name: str = 'ColorizeArtistic_gen'

        if self._model_artistic is not None and render_factor == self._render_factor_artistic:
            return  # nothing to do

        learn = gen_inference_deep(root_folder=root_folder, weights_name=weights_name)
        filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)

        self._model_artistic = ModelImageVisualizer(filtr)
        self._render_factor_artistic = render_factor

    def get_transformed_image(self, img_orig: Image, post_process: bool = True) -> Image:

        img_video = self._model_video.get_transformed_image(img_orig, self._render_factor_video,
                                                            post_process=post_process)
        if self._modelname == 'video':
            return img_video

        if self._modelname == 'stable':
            img_stable = self._model_stable.get_transformed_image(img_orig, self._render_factor_stable,
                                                                  post_process=post_process)
            # img_m = img * (1.0 - w) + img_video * weight
            return Image.blend(img_stable, img_video, self._video_weight)

        if self._modelname == 'artistic':
            img_artistic = self._model_artistic.get_transformed_image(img_orig, self._render_factor_artistic,
                                                                  post_process=post_process)
            # img_m = img * (1.0 - w) + img_video * weight
            return Image.blend(img_artistic, img_video, self._video_weight)

        return img_orig   # should never happen
