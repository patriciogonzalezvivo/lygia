/*
original_author: Patricio Gonzalez Vivo
description: Converts a RGB normal map into normal vectors
use: normalMap(<sampler2D> texture, <vec2> st, <vec2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - NORMALMAP_Z: Steepness of z before normalization, defaults to .01
    - NORMALMAP_SAMPLER_FNC(POS_UV): Function used to sample into the normal map texture, defaults to texture2D(tex,POS_UV).r
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef NORMALMAP_Z
#define NORMALMAP_Z .01
#endif

#ifndef NORMALMAP_SAMPLER_FNC
#define NORMALMAP_SAMPLER_FNC(POS_UV) SAMPLER_FNC(tex,POS_UV).r
#endif

#ifndef FNC_NORMALMAP
#define FNC_NORMALMAP
vec3 normalMap(sampler2D tex, vec2 st, vec2 pixel) {
    float center      = NORMALMAP_SAMPLER_FNC(st);
    float topLeft     = NORMALMAP_SAMPLER_FNC(st - pixel);
    float left        = NORMALMAP_SAMPLER_FNC(st - vec2(pixel.x, .0));
    float bottomLeft  = NORMALMAP_SAMPLER_FNC(st + vec2(-pixel.x, pixel.y));
    float top         = NORMALMAP_SAMPLER_FNC(st - vec2(.0, pixel.y));
    float bottom      = NORMALMAP_SAMPLER_FNC(st + vec2(.0, pixel.y));
    float topRight    = NORMALMAP_SAMPLER_FNC(st + vec2(pixel.x, -pixel.y));
    float right       = NORMALMAP_SAMPLER_FNC(st + vec2(pixel.x, .0));
    float bottomRight = NORMALMAP_SAMPLER_FNC(st + pixel);
    
    float dX = topRight + 2. * right + bottomRight - topLeft - 2. * left - bottomLeft;
    float dY = bottomLeft + 2. * bottom + bottomRight - topLeft - 2. * top - topRight;

    return normalize(vec3(dX, dY, NORMALMAP_Z) );
}
#endif