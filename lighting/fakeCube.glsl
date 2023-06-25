#include "../math/powFast.glsl"
#include "material/shininess.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: creates a fake cube and returns the value giving a normal direction
use: <vec3> fakeCube(<vec3> _normal [, <float> _shininnes])
options:
    - FAKECUBE_LIGHT_AMOUNT: amount of light to fake
    - FAKECUBE_ONLYXWALL: only the x wall is lit
    - FAKECUBE_ONLYYWALL: only the y wall is lit
    - FAKECUBE_ONLYZWALL: only the z wall is lit
    - FAKECUBE_NOFLOOR: removes the floor from the fake cube
    - FAKECUBE_NOROOF: removes the floor from the fake cube
    - FAKECUBE_NOXWALL: removes the x wall from the fake cube
    - FAKECUBE_NONXWALL: removes the -x wall from the fake cube
    - FAKECUBE_NOZWALL: removes the z wall from the fake cube
    - FAKECUBE_NOMZWALL: removes the -z wall from the fake cube
    - FAKECUBE_SAMPLE_FNC(UV): function to sample the fake cube
*/

#ifndef FAKECUBE_LIGHT_AMOUNT
#define FAKECUBE_LIGHT_AMOUNT 0.005
#endif

#ifndef FNC_FAKECUBE
#define FNC_FAKECUBE

vec3 fakeCube(const in vec3 _normal, const in float _shininnes) {

    #if defined(FAKECUBE_SAMPLE_FNC)
    vec3 colx = FAKECUBE_SAMPLE_FNC(d.yz).rgb;
    vec3 coly = FAKECUBE_SAMPLE_FNC(d.zx).rgb;
    vec3 colz = FAKECUBE_SAMPLE_FNC(d.xy).rgb;
    vec3 n = d*d;
    return (colx*n.x + coly*n.y + colz*n.z)/(n.x+n.y+n.z);

    #elif defined(FAKECUBE_ONLYXWALL)
    return vec3( powFast(saturate(_normal.x) + FAKECUBE_LIGHT_AMOUNT, _shininnes) );

    #elif defined(FAKECUBE_ONLYYWALL)
    return vec3( powFast(saturate(_normal.y) + FAKECUBE_LIGHT_AMOUNT, _shininnes) );

    #elif defined(FAKECUBE_ONLYZWALL)
    return vec3( powFast(saturate(_normal.z) + FAKECUBE_LIGHT_AMOUNT, _shininnes) );

    #else
    vec3 rAbs = abs(_normal);
    return vec3( powFast(max(max(rAbs.x, rAbs.y), rAbs.z) + FAKECUBE_LIGHT_AMOUNT, _shininnes)
        #if defined(FAKECUBE_NOFLOOR)
        * smoothstep(-1.0, 0., _normal.y) 
        #endif

        #if defined(FAKECUBE_NOROOF)
        * smoothstep(1.0, 0., _normal.y) 
        #endif

        #if defined(FAKECUBE_NOXWALL)
        * smoothstep(1.0, 0.0, _normal.x) 
        #endif

        #if defined(FAKECUBE_NONXWALL)
        * smoothstep(-1.0, 0., _normal.x) 
        #endif

        #if defined(FAKECUBE_NOZWALL)
        * smoothstep(-1.0, 0., _normal.z) 
        #endif

        #if defined(FAKECUBE_NONZWALL)
        * smoothstep(1.0, 0., _normal.z) 
        #endif
    );

    #endif
}

vec3 fakeCube(const in vec3 _normal) {
    return fakeCube(_normal, materialShininess() );
}

#endif