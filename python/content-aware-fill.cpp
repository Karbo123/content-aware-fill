/*
 * Python Binding for Content Aware Fill
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sstream>

#include "imageSynth.h"

namespace py = pybind11;

using Array = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>; // uint8, row-major

static const char *enums[] = {
    [IMAGE_SYNTH_SUCCESS] = "IMAGE_SYNTH_SUCCESS",
    [IMAGE_SYNTH_ERROR_INVALID_IMAGE_FORMAT] = "IMAGE_SYNTH_ERROR_INVALID_IMAGE_FORMAT",
    [IMAGE_SYNTH_ERROR_IMAGE_MASK_MISMATCH] = "IMAGE_SYNTH_ERROR_IMAGE_MASK_MISMATCH",
    [IMAGE_SYNTH_ERROR_PATCH_SIZE_EXCEEDED] = "IMAGE_SYNTH_ERROR_PATCH_SIZE_EXCEEDED",
    [IMAGE_SYNTH_ERROR_MATCH_CONTEXT_TYPE_RANGE] = "IMAGE_SYNTH_ERROR_MATCH_CONTEXT_TYPE_RANGE",
    [IMAGE_SYNTH_ERROR_EMPTY_TARGET] = "IMAGE_SYNTH_ERROR_EMPTY_TARGET",
    [IMAGE_SYNTH_ERROR_EMPTY_CORPUS] = "IMAGE_SYNTH_ERROR_EMPTY_CORPUS",
};


/*
 * Perform Content Aware Fill for `image` given the inpainting mask `mask`
 *
 * Params:
 *  - image (numpy.ndarray): shape=(H, W, C), uint8, C==1/2/3/4
 *  - mask (numpy.ndarray): shape=(H, W), uint8, nonzeros for regions to be filled
 * Return:
 *  - image_out (numpy.ndarray): shape=(H, W, C), uint8, C==1/2/3/4
 *  - state (str): a string indicating whether it succeeded (i.e. "IMAGE_SYNTH_SUCCESS")
 *
 */
std::tuple<Array, std::string> content_aware_fill(
    Array image, Array mask,
    bool isMakeSeamlesslyTileableHorizontally=false,
    bool isMakeSeamlesslyTileableVertically=false,
    int matchContextType=1,
    float mapWeight=0.5f,
    float sensitivityToOutliers=0.117f,
    int patchSize=30,
    int maxProbeCount=200
) {
    auto buff_image = image.request();
    auto buff_mask = mask.request();

    // check
    if (buff_image.ndim != 3 || 
        buff_mask .ndim != 2 ||
        buff_image.shape[0] != buff_mask.shape[0] || 
        buff_image.shape[1] != buff_mask.shape[1]
    ) {
        throw std::runtime_error("`image` must have shape == (H, W, 3), `mask` must have shape == (H, W)!");
    }
    if (image.flags() & py::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_ == 0 ||
        mask .flags() & py::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_ == 0
    ) {
        throw std::runtime_error("`image` and `mask` must be row-major and contiguous!");
    }

    auto image_out = Array(buff_image.shape);
    auto buff_out = image_out.request();
    auto image_out_ptr = static_cast<uint8_t*>(buff_out.ptr);
    memcpy(static_cast<void*>(image_out_ptr), static_cast<void*>(buff_image.ptr), image_out.nbytes());

    auto H = buff_image.shape[0];
    auto W = buff_image.shape[1];
    auto C = buff_image.shape[2];

    ImageBuffer struct_image = {
        .data = image_out_ptr,
        .width = W,
        .height = H,
        .rowBytes = W * C,
    };

    ImageBuffer struct_mask = {
        .data = static_cast<uint8_t*>(buff_mask.ptr),
        .width = W,
        .height = H,
        .rowBytes = W,
    };

    TImageSynthParameters params = {
        .isMakeSeamlesslyTileableHorizontally = int(isMakeSeamlesslyTileableHorizontally),
        .isMakeSeamlesslyTileableVertically = int(isMakeSeamlesslyTileableVertically),
        .matchContextType = matchContextType,
        .mapWeight = mapWeight,
        .sensitivityToOutliers = sensitivityToOutliers,
        .patchSize = patchSize,
        .maxProbeCount = maxProbeCount,
    };

    auto color_type = T_RGB;
    switch (C) {
        case 1: color_type = T_Gray;  break;
        case 2: color_type = T_GrayA; break;
        case 3: color_type = T_RGB;   break;
        case 4: color_type = T_RGBA;  break;
        default:
            std::stringstream ss;
            ss << "Number of channel must be 1(Gray), 2(GrayA), 3(RGB) or 4(RGBA), but found #channel=" << C << "!" << std::endl;
            throw std::runtime_error(ss.str());
    }

    int error = 0;
    int cancel = 0;
    error = imageSynth(&struct_image, &struct_mask, color_type, &params, NULL, NULL, &cancel);

    // return
    auto state = std::string(enums[error]);
    return {image_out, state};
}

PYBIND11_MODULE(libcwf, m) {
    m.doc() = "Content Aware Fill Algorithm";

    m.def("content_aware_fill", &content_aware_fill, 
          "Content Aware Fill Algorithm",
           py::arg("image"),
           py::arg("mask"),
           py::arg("isMakeSeamlesslyTileableHorizontally")=false,
           py::arg("isMakeSeamlesslyTileableVertically")=false,
           py::arg("matchContextType")=1,
           py::arg("mapWeight")=0.5f,
           py::arg("sensitivityToOutliers")=0.117f,
           py::arg("patchSize")=30,
           py::arg("maxProbeCount")=200
        );
}
