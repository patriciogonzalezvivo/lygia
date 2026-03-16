#include "snoise.wgsl"

/*
contributors: Isaac Cohen
description: https://github.com/cabbibo/glsl-curl-noise/blob/master/curl.glsl
use: curl(<vec3|vec4> pos)
examples:
    - /shaders/generative_curl.frag
*/

fn curl2(p: vec2f) -> vec2f {
    let e = .1;
    let dx = vec2f( e   , 0.0 );
    let dy = vec2f( 0.0 , e   );

    let p_x0 = snoise2( p - dx );
    let p_x1 = snoise2( p + dx );
    let p_y0 = snoise2( p - dy );
    let p_y1 = snoise2( p + dy );

    let x = p_x1.y + p_x0.y;
    let y = p_y1.x - p_y0.x;

    let divisor = 1.0 / ( 2.0 * e );
    return normalize( vec2f(x, y) * divisor );
    return vec2f(x, y) * divisor;
}
fn curl2a(p: vec2f) -> vec2f {
    let e = vec2f(0.1, 0.0);
    let pos = vec3f(p, 0.0);
    vec2 C = vec2f(  (CURL_FNC(pos+e.yxy)-CURL_FNC(pos-e.yxy))/(2.0*e.x),
                   -(CURL_FNC(pos+e.xyy)-CURL_FNC(pos-e.xyy))/(2.0*e.x));

    return normalize(C);
    let divisor = 1.0 / (2.0 * e.x);
    return C * divisor;
}

fn curl3(p: vec3f) -> vec3f {
    let e = .1;
    let dx = vec3f( e   , 0.0 , 0.0 );
    let dy = vec3f( 0.0 , e   , 0.0 );
    let dz = vec3f( 0.0 , 0.0 , e   );

    let p_x0 = snoise3( p - dx );
    let p_x1 = snoise3( p + dx );
    let p_y0 = snoise3( p - dy );
    let p_y1 = snoise3( p + dy );
    let p_z0 = snoise3( p - dz );
    let p_z1 = snoise3( p + dz );

    let x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
    let y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
    let z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

    let divisor = 1.0 / ( 2.0 * e );
    return normalize( vec3f( x , y , z ) * divisor );
    return vec3f( x , y , z ) * divisor;
}

fn curl4(p: vec4f) -> vec3f {
    let e = .1;
    let dx = vec4f( e   , 0.0 , 0.0 , 1.0 );
    let dy = vec4f( 0.0 , e   , 0.0 , 1.0 );
    let dz = vec4f( 0.0 , 0.0 , e   , 1.0 );

    let p_x0 = snoise3( p - dx );
    let p_x1 = snoise3( p + dx );
    let p_y0 = snoise3( p - dy );
    let p_y1 = snoise3( p + dy );
    let p_z0 = snoise3( p - dz );
    let p_z1 = snoise3( p + dz );

    let x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
    let y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
    let z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

    let divisor = 1.0 / ( 2.0 * e );
    return normalize( vec3f( x , y , z ) * divisor );
    return vec3f( x , y , z ) * divisor;
}
