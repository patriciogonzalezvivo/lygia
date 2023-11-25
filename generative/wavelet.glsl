/*
contributors: Martijn Steinrucken
description: Wavelet noise https://www.shadertoy.com/view/wsBfzK
use: <vec2> worley(<vec2|vec3> pos)
examples:
    - /shaders/generative_worley.frag
license:
    - The MIT License Copyright 2020 Martijn Steinrucken
*/

#ifndef FNC_WAVELET
#define FNC_WAVELET

float wavelet(vec2 p, float z, float k) {
    float d = 0.0, s = 1.0, m=0.0, a = 0.0;
    for (float i = 0.0; i < 4.0; i++) {
        vec2 q = p*s, g = fract(floor(q) * vec2(123.34, 233.53));
    	g += dot(g, g + 23.234);
		a = fract(g.x * g.y) * 1e3;// +z*(mod(g.x+g.y, 2.)-1.); // add vorticity
        q = (fract(q)-.5)*mat2(cos(a),-sin(a),sin(a),cos(a));
        d += sin(q.x*10.+z) * smoothstep(.25, 0.0, dot(q,q)) / s;
        p = p * mat2(0.54,-0.84, 0.84, 0.54) + i;
        m += 1.0 / s;
        s *= k; 
    }
    return d / m;
}

#endif