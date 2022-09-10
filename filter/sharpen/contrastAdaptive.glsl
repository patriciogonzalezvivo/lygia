
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
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SHARPEN_KERNELSIZE: Defaults 2
    - SHARPENCONTRASTADAPTIVE_TYPE: defaults to vec3
    - SHARPENCONTRASTADAPTIVE_SAMPLER_FNC(POS_UV): defaults to texture2D(tex, POS_UV).rgb
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

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef SHARPENCONTRASTADAPTIVE_TYPE
#ifdef SHARPEN_TYPE
#define SHARPENCONTRASTADAPTIVE_TYPE SHARPEN_TYPE
#else
#define SHARPENCONTRASTADAPTIVE_TYPE vec3
#endif
#endif

#ifndef SHARPENCONTRASTADAPTIVE_SAMPLER_FNC
#ifdef SHARPEN_SAMPLER_FNC
#define SHARPENCONTRASTADAPTIVE_SAMPLER_FNC(POS_UV) SHARPEN_SAMPLER_FNC(POS_UV)
#else
#define SHARPENCONTRASTADAPTIVE_SAMPLER_FNC(POS_UV) SAMPLER_FNC(tex, POS_UV).rgb
#endif
#endif

#ifndef FNC_SHARPENADCONTRASTAPTIVE
#define FNC_SHARPENADCONTRASTAPTIVE
SHARPENCONTRASTADAPTIVE_TYPE sharpenContrastAdaptive(sampler2D tex, vec2 st, vec2 pixel, float strenght) {
    float peak = -1.0 / mix(8.0, 5.0, saturate(strenght));
    
    // fetch a 3x3 neighborhood around the pixel 'e',
    //  a b c
    //  d(e)f
    //  g h i
    SHARPENCONTRASTADAPTIVE_TYPE a = SHARPENCONTRASTADAPTIVE_SAMPLER_FNC(st + vec2(-1., -1.) * pixel);
    SHARPENCONTRASTADAPTIVE_TYPE b = SHARPENCONTRASTADAPTIVE_SAMPLER_FNC(st + vec2( 0., -1.) * pixel);
    SHARPENCONTRASTADAPTIVE_TYPE c = SHARPENCONTRASTADAPTIVE_SAMPLER_FNC(st + vec2( 1., -1.) * pixel);
    SHARPENCONTRASTADAPTIVE_TYPE d = SHARPENCONTRASTADAPTIVE_SAMPLER_FNC(st + vec2(-1.,  0.) * pixel);
    SHARPENCONTRASTADAPTIVE_TYPE e = SHARPENCONTRASTADAPTIVE_SAMPLER_FNC(st + vec2( 0.,  0.) * pixel);
    SHARPENCONTRASTADAPTIVE_TYPE f = SHARPENCONTRASTADAPTIVE_SAMPLER_FNC(st + vec2( 1.,  0.) * pixel);
    SHARPENCONTRASTADAPTIVE_TYPE g = SHARPENCONTRASTADAPTIVE_SAMPLER_FNC(st + vec2(-1.,  1.) * pixel);
    SHARPENCONTRASTADAPTIVE_TYPE h = SHARPENCONTRASTADAPTIVE_SAMPLER_FNC(st + vec2( 0.,  1.) * pixel);
    SHARPENCONTRASTADAPTIVE_TYPE i = SHARPENCONTRASTADAPTIVE_SAMPLER_FNC(st + vec2( 1.,  1.) * pixel);

	// Soft min and max.
	//	a b c			  b
	//	d e f * 0.5	 +	d e f * 0.5
	//	g h i			  h
	// These are 2.0x bigger (factored out the extra multiply).
    SHARPENCONTRASTADAPTIVE_TYPE mnRGB = min(min(min(d, e), min(f, b)), h);
    SHARPENCONTRASTADAPTIVE_TYPE mnRGB2 = min(mnRGB, min(min(a, c), min(g, i)));
    mnRGB += mnRGB2;

    SHARPENCONTRASTADAPTIVE_TYPE mxRGB = max(max(max(d, e), max(f, b)), h);
    SHARPENCONTRASTADAPTIVE_TYPE mxRGB2 = max(mxRGB, max(max(a, c), max(g, i)));
    mxRGB += mxRGB2;

	// Smooth minimum distance to signal limit divided by smooth max.
	SHARPENCONTRASTADAPTIVE_TYPE ampRGB = saturate(min(mnRGB, 2.0 - mxRGB) / mxRGB);
	
	// Shaping amount of sharpening.
	SHARPENCONTRASTADAPTIVE_TYPE wRGB = sqrt(ampRGB) * peak;
	
	// Filter shape.
	//  0 w 0
	//  w 1 w
	//  0 w 0  
	SHARPENCONTRASTADAPTIVE_TYPE weightRGB = 1.0 + 4.0 * wRGB;
	SHARPENCONTRASTADAPTIVE_TYPE window = (b + d) + (f + h);
	return saturate((window * wRGB + e) / weightRGB);
}

SHARPENCONTRASTADAPTIVE_TYPE sharpenContrastAdaptive(sampler2D tex, vec2 st, vec2 pixel) {
    return sharpenContrastAdaptive(tex, st, pixel, 1.0);
}
#endif