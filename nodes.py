import numpy as np
import cv2
from tqdm import trange

try:
    from cv2.ximgproc import guidedFilter
except ImportError:
    print("\033[33mUnable to import guidedFilter, make sure you have only opencv-contrib-python or run the import_error_install.bat script\033[m")

import node_helpers
from comfy.utils import ProgressBar
from comfy_extras.nodes_post_processing import gaussian_kernel
from .raft import *

MAX_RESOLUTION=8192

# gaussian blur a tensor image batch in format [B x H x W x C] on H/W (spatial, per-image, per-channel)
def cv_blur_tensor(images, dx, dy):
    if min(dx, dy) > 100:
        np_img = torch.nn.functional.interpolate(images.detach().clone().movedim(-1,1), scale_factor=0.1, mode='bilinear').movedim(1,-1).cpu().numpy()
        for index, image in enumerate(np_img):
            np_img[index] = cv2.GaussianBlur(image, (dx // 20 * 2 + 1, dy // 20 * 2 + 1), 0)
        return torch.nn.functional.interpolate(torch.from_numpy(np_img).movedim(-1,1), size=(images.shape[1], images.shape[2]), mode='bilinear').movedim(1,-1)
    else:
        np_img = images.detach().clone().cpu().numpy()
        for index, image in enumerate(np_img):
            np_img[index] = cv2.GaussianBlur(image, (dx, dy), 0)
        return torch.from_numpy(np_img)

# guided filter a tensor image batch in format [B x H x W x C] on H/W (spatial, per-image, per-channel)
def guided_filter_tensor(ref, images, d, s):
    if d > 100:
        np_img = torch.nn.functional.interpolate(images.detach().clone().movedim(-1,1), scale_factor=0.1, mode='bilinear').movedim(1,-1).cpu().numpy()
        np_ref = torch.nn.functional.interpolate(ref.detach().clone().movedim(-1,1), scale_factor=0.1, mode='bilinear').movedim(1,-1).cpu().numpy()
        for index, image in enumerate(np_img):
            np_img[index] = guidedFilter(np_ref[index], image, d // 20 * 2 + 1, s)
        return torch.nn.functional.interpolate(torch.from_numpy(np_img).movedim(-1,1), size=(images.shape[1], images.shape[2]), mode='bilinear').movedim(1,-1)
    else:
        np_img = images.detach().clone().cpu().numpy()
        np_ref = ref.cpu().numpy()
        for index, image in enumerate(np_img):
            np_img[index] = guidedFilter(np_ref[index], image, d, s)
        return torch.from_numpy(np_img)

# std_dev and mean of tensor t within local spatial filter size d, per-image, per-channel [B x H x W x C]
def std_mean_filter(t, d):
    t_mean = cv_blur_tensor(t, d, d)
    t_diff_squared = (t - t_mean) ** 2
    t_std = torch.sqrt(cv_blur_tensor(t_diff_squared, d, d))
    return t_std, t_mean

def RGB2YCbCr(t):
    YCbCr = t.detach().clone()
    YCbCr[:,:,:,0] = 0.2123 * t[:,:,:,0] + 0.7152 * t[:,:,:,1] + 0.0722 * t[:,:,:,2]
    YCbCr[:,:,:,1] = 0 - 0.1146 * t[:,:,:,0] - 0.3854 * t[:,:,:,1] + 0.5 * t[:,:,:,2]
    YCbCr[:,:,:,2] = 0.5 * t[:,:,:,0] - 0.4542 * t[:,:,:,1] - 0.0458 * t[:,:,:,2]
    return YCbCr

def YCbCr2RGB(t):
    RGB = t.detach().clone()
    RGB[:,:,:,0] = t[:,:,:,0] + 1.5748 * t[:,:,:,2]
    RGB[:,:,:,1] = t[:,:,:,0] - 0.1873 * t[:,:,:,1] - 0.4681 * t[:,:,:,2]
    RGB[:,:,:,2] = t[:,:,:,0] + 1.8556 * t[:,:,:,1]
    return RGB

def hsv_to_rgb(h, s, v):
    if s:
        if h == 1.0: h = 0.0
        i = int(h*6.0)
        f = h*6.0 - i

        w = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        if i==0: return (v, t, w)
        if i==1: return (q, v, w)
        if i==2: return (w, v, t)
        if i==3: return (w, q, v)
        if i==4: return (t, w, v)
        if i==5: return (v, w, q)
    else: return (v, v, v)

def sRGBtoLinear(npArray):
    less = npArray <= 0.0404482362771082
    npArray[less] = npArray[less] / 12.92
    npArray[~less] = np.power((npArray[~less] + 0.055) / 1.055, 2.4)

def linearToSRGB(npArray):
    less = npArray <= 0.0031308
    npArray[less] = npArray[less] * 12.92
    npArray[~less] = np.power(npArray[~less], 1/2.4) * 1.055 - 0.055

def linearToTonemap(npArray, tonemap_scale):
    npArray /= tonemap_scale
    more = npArray > 0.06
    SLog3 = np.clip((np.log10((npArray + 0.01)/0.19) * 261.5 + 420) / 1023, 0, 1)
    npArray[more] = np.power(1 / (1 + (1 / np.power(SLog3[more] / (1 - SLog3[more]), 1.7))), 1.7)
    npArray *= tonemap_scale

def tonemapToLinear(npArray, tonemap_scale):
    npArray /= tonemap_scale
    more = npArray > 0.06
    x = np.power(np.clip(npArray, 0.000001, 1), 1/1.7)
    ut = 1 / (1 + np.power((-1 / x) * (x - 1), 1/1.7))
    npArray[more] = np.power(10, (ut[more] * 1023 - 420)/261.5) * 0.19 - 0.01
    npArray *= tonemap_scale

def exposure(npArray, stops):
    more = npArray > 0
    npArray[more] *= pow(2, stops)



class BetterFilmGrain:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 2.0, "step": 0.05}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "toe": ("FLOAT", {"default": 0.0, "min": -0.2, "max": 0.5, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "grain"

    CATEGORY = "image/filters"

    def grain(self, image, scale, strength, saturation, toe, seed):
        t = image.detach().clone()
        torch.manual_seed(seed)
        grain = torch.rand(t.shape[0], int(t.shape[1] // scale), int(t.shape[2] // scale), 3)
        
        YCbCr = RGB2YCbCr(grain)
        YCbCr[:,:,:,0] = cv_blur_tensor(YCbCr[:,:,:,0], 3, 3)
        YCbCr[:,:,:,1] = cv_blur_tensor(YCbCr[:,:,:,1], 15, 15)
        YCbCr[:,:,:,2] = cv_blur_tensor(YCbCr[:,:,:,2], 11, 11)
        
        grain = (YCbCr2RGB(YCbCr) - 0.5) * strength
        grain[:,:,:,0] *= 2
        grain[:,:,:,2] *= 3
        grain += 1
        grain = grain * saturation + grain[:,:,:,1].unsqueeze(3).repeat(1,1,1,3) * (1 - saturation)
        
        grain = torch.nn.functional.interpolate(grain.movedim(-1,1), size=(t.shape[1], t.shape[2]), mode='bilinear').movedim(1,-1)
        t[:,:,:,:3] = torch.clip((1 - (1 - t[:,:,:,:3]) * grain) * (1 - toe) + toe, 0, 1)
        return(t,)

NODE_CLASS_MAPPINGS = {
    # "AdainFilterLatent": AdainFilterLatent,
    # "AdainImage": AdainImage,
    # "AdainLatent": AdainLatent,
    # "AlphaClean": AlphaClean,
    # "AlphaMatte": AlphaMatte,
    # "BatchAlign": BatchAlign,
    # "BatchAverageImage": BatchAverageImage,
    # "BatchAverageUnJittered": BatchAverageUnJittered,
    # "BatchNormalizeImage": BatchNormalizeImage,
    # "BatchNormalizeLatent": BatchNormalizeLatent,
    "BetterFilmGrain": BetterFilmGrain,
    # "BilateralFilterImage": BilateralFilterImage,
    # "BlurImageFast": BlurImageFast,
    # "BlurMaskFast": BlurMaskFast,
    # "ClampImage": ClampImage,
    # "ClampOutliers": ClampOutliers,
    # "ColorMatchImage": ColorMatchImage,
    # "ConditioningSubtract": ConditioningSubtract,
    # "ConvertNormals": ConvertNormals,
    # "CustomNoise": CustomNoise,
    # "DepthToNormals": DepthToNormals,
    # "DifferenceChecker": DifferenceChecker,
    # "DilateErodeMask": DilateErodeMask,
    # "EnhanceDetail": EnhanceDetail,
    # "ExposureAdjust": ExposureAdjust,
    # "ExtractNFrames": ExtractNFrames,
    # "FrequencyCombine": FrequencyCombine,
    # "FrequencySeparate": FrequencySeparate,
    # "GameOfLife": GameOfLife,
    # "GuidedFilterAlpha": GuidedFilterAlpha,
    # "GuidedFilterImage": GuidedFilterImage,
    # "ImageConstant": ImageConstant,
    # "ImageConstantHSV": ImageConstantHSV,
    # "InpaintConditionApply": InpaintConditionApply,
    # "InpaintConditionEncode": InpaintConditionEncode,
    # "InstructPixToPixConditioningAdvanced": InstructPixToPixConditioningAdvanced,
    # "JitterImage": JitterImage,
    # "Keyer": Keyer,
    # "LatentNormalizeShuffle": LatentNormalizeShuffle,
    # "LatentStats": LatentStats,
    # "MedianFilterImage": MedianFilterImage,
    # "MergeFramesByIndex": MergeFramesByIndex,
    # "ModelTest": ModelTest,
    # "NormalMapSimple": NormalMapSimple,
    # "OffsetLatentImage": OffsetLatentImage,
    # "PrintSigmas": PrintSigmas,
    # "RelightSimple": RelightSimple,
    # "RemapRange": RemapRange,
    # "RestoreDetail": RestoreDetail,
    # "SharpenFilterLatent": SharpenFilterLatent,
    # "ShuffleChannels": ShuffleChannels,
    # "Tonemap": Tonemap,
    # "UnJitterImage": UnJitterImage,
    # "UnTonemap": UnTonemap,
    # "VisualizeLatents": VisualizeLatents,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # "AdainFilterLatent": "AdaIN Filter (Latent)",
    # "AdainImage": "AdaIN (Image)",
    # "AdainLatent": "AdaIN (Latent)",
    # "AlphaClean": "Alpha Clean",
    # "AlphaMatte": "Alpha Matte",
    # "BatchAlign": "Batch Align (RAFT)",
    # "BatchAverageImage": "Batch Average Image",
    # "BatchAverageUnJittered": "Batch Average Un-Jittered",
    # "BatchNormalizeImage": "Batch Normalize (Image)",
    # "BatchNormalizeLatent": "Batch Normalize (Latent)",
    "BetterFilmGrain": "Better Film Grain",
    # "BilateralFilterImage": "Bilateral Filter Image",
    # "BlurImageFast": "Blur Image (Fast)",
    # "BlurMaskFast": "Blur Mask (Fast)",
    # "ClampImage": "Clamp Image",
    # "ClampOutliers": "Clamp Outliers",
    # "ColorMatchImage": "Color Match Image",
    # "ConditioningSubtract": "ConditioningSubtract",
    # "ConvertNormals": "Convert Normals",
    # "CustomNoise": "CustomNoise",
    # "DepthToNormals": "Depth To Normals",
    # "DifferenceChecker": "Difference Checker",
    # "DilateErodeMask": "Dilate/Erode Mask",
    # "EnhanceDetail": "Enhance Detail",
    # "ExposureAdjust": "Exposure Adjust",
    # "ExtractNFrames": "Extract N Frames",
    # "FrequencyCombine": "Frequency Combine",
    # "FrequencySeparate": "Frequency Separate",
    # "GameOfLife": "Game Of Life",
    # "GuidedFilterAlpha": "(DEPRECATED) Guided Filter Alpha",
    # "GuidedFilterImage": "Guided Filter Image",
    # "ImageConstant": "Image Constant Color (RGB)",
    # "ImageConstantHSV": "Image Constant Color (HSV)",
    # "InpaintConditionApply": "Inpaint Condition Apply",
    # "InpaintConditionEncode": "Inpaint Condition Encode",
    # "InstructPixToPixConditioningAdvanced": "InstructPixToPixConditioningAdvanced",
    # "JitterImage": "Jitter Image",
    # "Keyer": "Keyer",
    # "LatentNormalizeShuffle": "LatentNormalizeShuffle",
    # "LatentStats": "Latent Stats",
    # "MedianFilterImage": "Median Filter Image",
    # "MergeFramesByIndex": "Merge Frames By Index",
    # "ModelTest": "Model Test",
    # "NormalMapSimple": "Normal Map (Simple)",
    # "OffsetLatentImage": "Offset Latent Image",
    # "PrintSigmas": "PrintSigmas",
    # "RelightSimple": "Relight (Simple)",
    # "RemapRange": "Remap Range",
    # "RestoreDetail": "Restore Detail",
    # "SharpenFilterLatent": "Sharpen Filter (Latent)",
    # "ShuffleChannels": "Shuffle Channels",
    # "Tonemap": "Tonemap",
    # "UnJitterImage": "Un-Jitter Image",
    # "UnTonemap": "UnTonemap",
    # "VisualizeLatents": "Visualize Latents",
}