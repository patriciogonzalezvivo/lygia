/*
description: Include all available color spaces conversions
*/

// CMYK
#include "space/cmyk2rgb.wgsl"
#include "space/rgb2cmyk.wgsl"

// Gamma
#include "space/gamma2linear.wgsl"
#include "space/linear2gamma.wgsl"

// HSV
#include "space/hsv2rgb.wgsl"
#include "space/rgb2hsv.wgsl"

// Kelvin
#include "space/k2rgb.wgsl"

// LAB
#include "space/rgb2lab.wgsl"
#include "space/lab2rgb.wgsl"

// LMS
#include "space/rgb2lms.wgsl"
#include "space/lms2rgb.wgsl"

// OkLab
#include "space/oklab2rgb.wgsl"
#include "space/rgb2oklab.wgsl"

// sRGB
#include "space/srgb2rgb.wgsl"
#include "space/rgb2srgb.wgsl"

// Wavelength
#include "space/w2rgb.wgsl"

// XYZ
#include "space/rgb2xyz.wgsl"
#include "space/xyz2rgb.wgsl"

// YCbCr
#include "space/YCbCr2rgb.wgsl"
#include "space/rgb2YCbCr.wgsl"

// YIQ
#include "space/yiq2rgb.wgsl"
#include "space/rgb2yiq.wgsl"

// YPbPr
#include "space/YPbPr2rgb.wgsl"
#include "space/rgb2YPbPr.wgsl"

// YUV
#include "space/yuv2rgb.wgsl"
#include "space/rgb2yuv.wgsl"
