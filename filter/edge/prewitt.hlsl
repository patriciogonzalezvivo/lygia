#include "../../sampler.hlsl"

/*
contributors: Brad Larson
description: Adapted version of Prewitt edge detection from https://github.com/BradLarson/GPUImage2
use: edgePrewitt(<SAMPLER_TYPE> texture, <float2> st, <float2> scale)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - EDGEPREWITT_TYPE: Return type, defaults to float
    - EDGEPREWITT_SAMPLER_FNC: Function used to sample the input texture, defaults to texture2D(tex,TEX, UV).r
*/

#ifndef EDGEPREWITT_TYPE
#ifdef EDGE_TYPE
#define EDGEPREWITT_TYPE EDGE_TYPE
#else
#define EDGEPREWITT_TYPE float
#endif
#endif

#ifndef EDGEPREWITT_SAMPLER_FNC
#ifdef EDGE_SAMPLER_FNC
#define EDGEPREWITT_SAMPLER_FNC(TEX, UV) EDGE_SAMPLER_FNC(TEX, UV)
#else
#define EDGEPREWITT_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).r
#endif
#endif

#ifndef FNC_EDGEPREWITT
#define FNC_EDGEPREWITT
EDGEPREWITT_TYPE edgePrewitt(in SAMPLER_TYPE tex, in float2 st, in float2 offset) {
    // get samples around pixel
    EDGEPREWITT_TYPE tleft = EDGEPREWITT_SAMPLER_FNC(tex, st + float2(-offset.x, offset.y));
    EDGEPREWITT_TYPE left = EDGEPREWITT_SAMPLER_FNC(tex, st + float2(-offset.x, 0.));
    EDGEPREWITT_TYPE bleft = EDGEPREWITT_SAMPLER_FNC(tex, st + float2(-offset.x, -offset.y));
    EDGEPREWITT_TYPE top = EDGEPREWITT_SAMPLER_FNC(tex, st + float2(0., offset.y));
    EDGEPREWITT_TYPE bottom = EDGEPREWITT_SAMPLER_FNC(tex, st + float2(0., -offset.y));
    EDGEPREWITT_TYPE tright = EDGEPREWITT_SAMPLER_FNC(tex, st + offset);
    EDGEPREWITT_TYPE right = EDGEPREWITT_SAMPLER_FNC(tex, st + float2(offset.x, 0.));
    EDGEPREWITT_TYPE bright = EDGEPREWITT_SAMPLER_FNC(tex, st + float2(offset.x, -offset.y));
    EDGEPREWITT_TYPE x = -tleft - top - tright + bleft + bottom + bright;
    EDGEPREWITT_TYPE y = -bleft - left - tleft + bright + right + tright;
    return sqrt((x * x) + (y * y));
}
#endif
