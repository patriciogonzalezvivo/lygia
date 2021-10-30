#include "../material.glsl"
#include "ao.glsl"
#include "normal.glsl"
#include "softShadow.glsl"
#include "cast.glsl"

/*
author: Patricio Gonzalez Vivo
description: Material Constructor. Designed to integrate with GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines 
use: 
    - void raymarchMaterial(in <vec3> ro, in <vec3> rd, out material _mat)
    - material raymarchMaterial(in <vec3> ro, in <vec3> rd)
    - LIGHT_POSITION: in glslViewer is u_light
license: |
    The MIT License
    Copyright © 2013 Inigo Quilez
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    https://www.youtube.com/c/InigoQuilez
    https://iquilezles.org/

    A list of useful distance function to simple primitives. All
    these functions (except for ellipsoid) return an exact
    euclidean distance, meaning they produce a better SDF than
    what you'd get if you were constructing them from boolean
    operations (such as cutting an infinite cylinder with two planes).

    List of other 3D SDFs:
       https://www.shadertoy.com/playlist/43cXRl
    and
       http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
*/

#ifndef LIGHT_POSITION
#if defined(GLSLVIEWER)
#define LIGHT_POSITION u_light
#else
#define LIGHT_POSITION vec3(0.0, 10.0, -50.0)
#endif
#endif

#ifndef FNC_RAYMARCHMATERIAL
#define FNC_RAYMARCHMATERIAL

void raymarchMaterial( in vec3 ro, in vec3 rd, inout Material mat) { 
    vec4 res = raymarchCast(ro, rd);

    vec3 col = vec3(0.0);
    vec3 m = res.rgb;
    float t = res.a;

    vec3 pos = ro + t * rd;
    vec3 nor = raymarchNormal( pos );
    float occ = raymarchAO( pos, nor );

    mat.baseColor = res;
    mat.normal = nor;
    mat.ambientOcclusion = occ;

    #if defined(SHADING_SHADOWS)
    vec3 ref = reflect( rd, nor );
    vec3 lig = normalize( LIGHT_POSITION );

    vec3  hal = normalize( lig-rd );
    float amb = clamp( 0.5+0.5*nor.y, 0.0, 1.0 );
    float dif = clamp( dot( nor, lig ), 0.0, 1.0 );
    float bac = clamp( dot( nor, normalize(vec3(-lig.x,0.0,-lig.z))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);
    float dom = smoothstep( -0.1, 0.1, ref.y );
    float fre = pow( clamp(1.0+dot(nor,rd),0.0,1.0), 2.0 );

    dif *= raymarchSoftShadow( pos, lig, 0.02, 2.5 );
    dom *= raymarchSoftShadow( pos, ref, 0.02, 2.5 );

    mat.shadows = 1.30 * dif;
    mat.shadows += 0.40 * amb * occ;
    mat.shadows += 0.50 * dom * occ;
    mat.shadows += 0.50 * bac * occ;
    mat.shadows += 0.25 * fre * occ * 0.25;
    #endif
}

Material raymarchMaterial( in vec3 ro, in vec3 rd) { 
    Material mat;
    raymarchMaterial( ro, rd, mat);
    return mat;
}

#endif