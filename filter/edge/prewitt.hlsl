/*
original_author: Brad Larson
description: Adapted version of Prewitt edge detection from https://github.com/BradLarson/GPUImage2
use: edgePrewitt(<sampler2D> texture, <float2> st, <float2> scale)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - EDGEPREWITT_TYPE: Return type, defaults to float
    - EDGEPREWITT_SAMPLER_FNC: Function used to sample the input texture, defaults to texture2D(tex,POS_UV).r
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) tex2D(TEX, UV)
#endif

#ifndef EDGEPREWITT_TYPE
#ifdef EDGE_TYPE
#define EDGEPREWITT_TYPE EDGE_TYPE
#else
#define EDGEPREWITT_TYPE float
#endif
#endif

#ifndef EDGEPREWITT_SAMPLER_FNC
#ifdef EDGE_SAMPLER_FNC
#define EDGEPREWITT_SAMPLER_FNC(POS_UV) EDGE_SAMPLER_FNC(POS_UV)
#else
#define EDGEPREWITT_SAMPLER_FNC(POS_UV) SAMPLER_FNC(tex, POS_UV).r
#endif
#endif

#ifndef FNC_EDGEPREWITT
#define FNC_EDGEPREWITT
EDGEPREWITT_TYPE edgePrewitt(in sampler2D tex, in float2 st, in float2 offset) {
    // get samples around pixel
    EDGEPREWITT_TYPE tleft = EDGEPREWITT_SAMPLER_FNC(st + float2(-offset.x, offset.y));
    EDGEPREWITT_TYPE left = EDGEPREWITT_SAMPLER_FNC(st + float2(-offset.x, 0.));
    EDGEPREWITT_TYPE bleft = EDGEPREWITT_SAMPLER_FNC(st + float2(-offset.x, -offset.y));
    EDGEPREWITT_TYPE top = EDGEPREWITT_SAMPLER_FNC(st + float2(0., offset.y));
    EDGEPREWITT_TYPE bottom = EDGEPREWITT_SAMPLER_FNC(st + float2(0., -offset.y));
    EDGEPREWITT_TYPE tright = EDGEPREWITT_SAMPLER_FNC(st + offset);
    EDGEPREWITT_TYPE right = EDGEPREWITT_SAMPLER_FNC(st + float2(offset.x, 0.));
    EDGEPREWITT_TYPE bright = EDGEPREWITT_SAMPLER_FNC(st + float2(offset.x, -offset.y));
    EDGEPREWITT_TYPE x = -tleft - top - tright + bleft + bottom + bright;
    EDGEPREWITT_TYPE y = -bleft - left - tleft + bright + right + tright;
    return sqrt((x * x) + (y * y));
}
#endif
