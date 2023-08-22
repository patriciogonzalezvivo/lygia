#include "..animation/easing/sineIn.glsl"
#include "../math/lengthSq.hlsl"

/*
original_author:  [shadertoyJiang, Kathy McGuiness]
description: |
    It returns a kaleidescope pattern.  Based on this [shader](https://www.shadertoy.com/view/ctByWz)

    Some notes about useage:

    * Higher values for zoom and iterations results in more detail
    * For a more classic kaleidescope effect, use m = 1.75 and n = 1.15;
    * For a "tie-dye" look, use m = 1.5 and n = 1.3
    * Calling the kaleidescope twice with different parameters creates some interesting effects
    * Clamping the returned value is useful if colors are to bright
use: kaleidescope(<float2> st, <float2> pixel, <float> t, <float> zoom, <float> m, <float> n, <int> interations)
*/

#ifndef FNC_KALEIDESCOPE
#define FNC_KALEIDESCOPE
float2 kaleidescope( float2 st, float2 pixel, float speed, float zoom, float m, float n, int N) {
    #ifdef CENTER_2D
    st -= CENTER_2D;
    #else
    st -= 0.5;
    #endif
    float3 r = float3(pixel.x, pixel.x, pixel.x);
    float3 uv = float3((2.0 * st.xyy - r)/r.x * zoom, (2.0 * st.xyy- r)/r.x * zoom,(2.0 * st.xyy- r)/r.x * zoom);
    uv.z = sineIn(t);
	uv *= 0.35;
    for (int i = 0; i< N; i++) {
		uv = abs( uv ) / lengthSq( uv ) - m; 
        uv = length( uv )*( uv + n);
	}
    uv.x = lengthSq(uv) * 0.5;
    return uv.xy;
}
#endif
