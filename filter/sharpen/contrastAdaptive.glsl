
#include "../../math/saturate.glsl"

/*
author: deus0ww
description: |
    Contrast Adaptive Sharpening - deus0ww - 2020-08-04 
    Orginal: https://github.com/GPUOpen-Effects/FidelityFX-CAS
    Reshade: https://gist.github.com/SLSNe/bbaf2d77db0b2a2a0755df581b3cf00c
    Reshade: https://gist.github.com/martymcmodding/30304c4bffa6e2bd2eb59ff8bb09d135
use: sharpenContrastAdaptive(<sampler2D> texture, <vec2> st, <vec2> pixel, <float> strenght)
options:
    SHARPEN_KERNELSIZE: Defaults 2
    SHARPEN_TYPE: defaults to vec3
    SHARPEN_SAMPLER_FNC(POS_UV): defaults to texture2D(tex, POS_UV).rgb
license: |
    Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
*/

#ifndef SHARPEN_TYPE
#define SHARPEN_TYPE vec3
#endif

#ifndef SHARPEN_SAMPLER_FNC
#define SHARPEN_SAMPLER_FNC(POS_UV) texture2D(tex, POS_UV).rgb
#endif

#ifndef FNC_SHARPENADCONTRASTAPTIVE
#define FNC_SHARPENADCONTRASTAPTIVE
SHARPEN_TYPE sharpenContrastAdaptive(sampler2D tex, vec2 st, vec2 pixel, float strenght) {
    float peak = -1.0 / mix(8.0, 5.0, saturate(strenght));
    
    // fetch a 3x3 neighborhood around the pixel 'e',
    //  a b c
    //  d(e)f
    //  g h i
    SHARPEN_TYPE a = SHARPEN_SAMPLER_FNC(st + vec2(-1., -1.) * pixel);
    SHARPEN_TYPE b = SHARPEN_SAMPLER_FNC(st + vec2( 0., -1.) * pixel);
    SHARPEN_TYPE c = SHARPEN_SAMPLER_FNC(st + vec2( 1., -1.) * pixel);
    SHARPEN_TYPE d = SHARPEN_SAMPLER_FNC(st + vec2(-1.,  0.) * pixel);
    SHARPEN_TYPE e = SHARPEN_SAMPLER_FNC(st + vec2( 0.,  0.) * pixel);
    SHARPEN_TYPE f = SHARPEN_SAMPLER_FNC(st + vec2( 1.,  0.) * pixel);
    SHARPEN_TYPE g = SHARPEN_SAMPLER_FNC(st + vec2(-1.,  1.) * pixel);
    SHARPEN_TYPE h = SHARPEN_SAMPLER_FNC(st + vec2( 0.,  1.) * pixel);
    SHARPEN_TYPE i = SHARPEN_SAMPLER_FNC(st + vec2( 1.,  1.) * pixel);

	// Soft min and max.
	//	a b c			  b
	//	d e f * 0.5	 +	d e f * 0.5
	//	g h i			  h
	// These are 2.0x bigger (factored out the extra multiply).
    SHARPEN_TYPE mnRGB = min(min(min(d, e), min(f, b)), h);
    SHARPEN_TYPE mnRGB2 = min(mnRGB, min(min(a, c), min(g, i)));
    mnRGB += mnRGB2;

    SHARPEN_TYPE mxRGB = max(max(max(d, e), max(f, b)), h);
    SHARPEN_TYPE mxRGB2 = max(mxRGB, max(max(a, c), max(g, i)));
    mxRGB += mxRGB2;

	// Smooth minimum distance to signal limit divided by smooth max.
	SHARPEN_TYPE ampRGB = saturate(min(mnRGB, 2.0 - mxRGB) / mxRGB);
	
	// Shaping amount of sharpening.
	SHARPEN_TYPE wRGB = sqrt(ampRGB) * peak;
	
	// Filter shape.
	//  0 w 0
	//  w 1 w
	//  0 w 0  
	SHARPEN_TYPE weightRGB = 1.0 + 4.0 * wRGB;
	SHARPEN_TYPE window = (b + d) + (f + h);
	return saturate((window * wRGB + e) / weightRGB);
}

SHARPEN_TYPE sharpenContrastAdaptive(sampler2D tex, vec2 st, vec2 pixel) {
    return sharpenContrastAdaptive(tex, st, pixel, 1.0);
}
#endif