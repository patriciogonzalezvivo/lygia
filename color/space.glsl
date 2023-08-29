/*
description: Include all available color spaces conversions
*/

// CMYK
#include "space/cmyk2rgb.glsl"
#include "space/rgb2cmyk.glsl"

// Gamma
#include "space/gamma2linear.glsl"
#include "space/linear2gamma.glsl"

// HSV
#include "space/hsv2rgb.glsl"
#include "space/rgb2hsv.glsl"

// Kelvin
#include "space/k2rgb.glsl"

// LAB
#include "space/rgb2lab.glsl"
#include "space/lab2rgb.glsl"

// LMS
#include "space/rgb2lms.glsl"
#include "space/lms2rgb.glsl"

// OkLab
#include "space/oklab2rgb.glsl"
#include "space/rgb2oklab.glsl"

// sRGB
#include "space/srgb2rgb.glsl"
#include "space/rgb2srgb.glsl"

// Wavelength
#include "space/w2rgb.glsl"

// XYZ
#include "space/rgb2xyz.glsl"
#include "space/xyz2rgb.glsl"

// YCbCr
#include "space/YCbCr2rgb.glsl"
#include "space/rgb2YCbCr.glsl"

// YIQ
#include "space/yiq2rgb.glsl"
#include "space/rgb2yiq.glsl"

// YPbPr
#include "space/YPbPr2rgb.glsl"
#include "space/rgb2YPbPr.glsl"

// YUV
#include "space/yuv2rgb.glsl"
#include "space/rgb2yuv.glsl"
