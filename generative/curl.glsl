#include "snoise.glsl"

/*
contributors: Isaac Cohen
description: https://github.com/cabbibo/glsl-curl-noise/blob/master/curl.glsl
use: curl(<vec3|vec4> pos)
examples:
    - /shaders/generative_curl.frag
*/

#ifndef FNC_CURL
#define FNC_CURL

#ifndef CURL_FNC
vec2 curl( vec2 p ) {
    const float e = .1;
    vec2 dx = vec2( e   , 0.0 );
    vec2 dy = vec2( 0.0 , e   );

    vec2 p_x0 = snoise2( p - dx );
    vec2 p_x1 = snoise2( p + dx );
    vec2 p_y0 = snoise2( p - dy );
    vec2 p_y1 = snoise2( p + dy );

    float x = p_x1.y + p_x0.y;
    float y = p_y1.x - p_y0.x;

    const float divisor = 1.0 / ( 2.0 * e );
    #ifndef CURL_UNNORMALIZED
    return normalize( vec2(x, y) * divisor );
    #else
    return vec2(x, y) * divisor;
    #endif
}
#else
vec2 curl( vec2 p ) {
    vec2 e = vec2(0.1, 0.0);
    vec3 pos = vec3(p, 0.0);
    vec2 C = vec2(  (CURL_FNC(pos+e.yxy)-CURL_FNC(pos-e.yxy))/(2.0*e.x),
                   -(CURL_FNC(pos+e.xyy)-CURL_FNC(pos-e.xyy))/(2.0*e.x));

    #ifndef CURL_UNNORMALIZED
    return normalize(C);
    #else
    float divisor = 1.0 / (2.0 * e.x);
    return C * divisor;
    #endif
}
#endif

vec3 curl( vec3 p ){
    const float e = .1;
    vec3 dx = vec3( e   , 0.0 , 0.0 );
    vec3 dy = vec3( 0.0 , e   , 0.0 );
    vec3 dz = vec3( 0.0 , 0.0 , e   );

    vec3 p_x0 = snoise3( p - dx );
    vec3 p_x1 = snoise3( p + dx );
    vec3 p_y0 = snoise3( p - dy );
    vec3 p_y1 = snoise3( p + dy );
    vec3 p_z0 = snoise3( p - dz );
    vec3 p_z1 = snoise3( p + dz );

    float x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
    float y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
    float z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

    const float divisor = 1.0 / ( 2.0 * e );
    #ifndef CURL_UNNORMALIZED
    return normalize( vec3( x , y , z ) * divisor );
    #else
    return vec3( x , y , z ) * divisor;
    #endif
}

vec3 curl( vec4 p ){
    const float e = .1;
    vec4 dx = vec4( e   , 0.0 , 0.0 , 1.0 );
    vec4 dy = vec4( 0.0 , e   , 0.0 , 1.0 );
    vec4 dz = vec4( 0.0 , 0.0 , e   , 1.0 );

    vec3 p_x0 = snoise3( p - dx );
    vec3 p_x1 = snoise3( p + dx );
    vec3 p_y0 = snoise3( p - dy );
    vec3 p_y1 = snoise3( p + dy );
    vec3 p_z0 = snoise3( p - dz );
    vec3 p_z1 = snoise3( p + dz );

    float x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
    float y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
    float z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

    const float divisor = 1.0 / ( 2.0 * e );
    #ifndef CURL_UNNORMALIZED
    return normalize( vec3( x , y , z ) * divisor );
    #else
    return vec3( x , y , z ) * divisor;
    #endif
}

#endif