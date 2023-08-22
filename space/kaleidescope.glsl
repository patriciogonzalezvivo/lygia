#include "..animation/easing/sineIn.glsl"
#include "../math/lengthSq.glsl"

/*
original_author:  [shadertoyJiang, Kathy McGuiness]
description: |
    It returns a kaleidescope pattern  Based on this [shader](https://www.shadertoy.com/view/ctByWz)

    Some notes about useage:

    * Higher values for zoom and iterations results in more detail
    * For a more classic kaleidescope effect, use uv.x, m = 1.75 and n = 1.15;
    * For a "tie-dye" look, use, uv.x, m = 1.5 and n = 1.3
    * Calling the kaleidescope twice with different parameters creates some interesting effects
    * Clamping the returned value is useful if colors are to bright
use: kaleidescope(<vec2> st, <vec2> pixel, <float> t, <float> zoom, <float> m, <float> n, <int> interations)
*/

#ifndef FNC_KALEIDESCOPE
#define FNC_KALEIDESCOPE
vec2 kaleidescope( vec2 st, vec2 pixel, float t, float zoom, float m, float n, int N) {
    #ifdef CENTER_2D
    st -= CENTER_2D;
    #else
    st -= 0.5;
    #endif
    vec3 r = vec3(pixel.x);
    vec3 uv = vec3((2.0 * st.xyy- r)/r.x * zoom);
    uv.z = sineIn(t);
    uv *= 0.35;
    #ifdef PLATFORM_WEBGL
    for (int i = 0; i< 10; i++) {
        if (i >= N) break;
    #else
    for (int i = 0; i< N; i++) {
    #endif
	uv = abs( uv ) / lengthSq( uv ) - m; 
        uv = length( uv )*( uv + n);
	}
    uv.x = lengthSq( uv ) * 0.5 ;
    return uv.xy;
}
#endif
