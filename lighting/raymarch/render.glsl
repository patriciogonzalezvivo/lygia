#include "cast.glsl"
#include "ao.glsl"
#include "normal.glsl"
#include "softShadow.glsl"
#include "../../math/saturate.glsl"

/*
author:  Inigo Quiles
description: raymarching renderer
use: <vec4> raymarchRender( in <vec3> ro, in <vec3> rd ) 
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
#define LIGHT_POSITION  u_light
#endif

#ifndef LIGHT_COLOR
#define LIGHT_COLOR     u_lightColor
#endif

#ifndef RAYMARCH_AMBIENT
#define RAYMARCH_AMBIENT vec3(1.0)
#endif

#ifndef RAYMARCH_BACKGROUND
#define RAYMARCH_BACKGROUND vec3(0.0)
// #define RAYMARCH_BACKGROUND ( vec3(0.7, 0.9, 1.0) +rd.y*0.8 )
#endif

#ifndef FNC_RAYMARCHRENDER
#define FNC_RAYMARCHRENDER

vec4 raymarchRender( in vec3 ro, in vec3 rd ) { 
    vec3 col = vec3(0.0);
    col = RAYMARCH_BACKGROUND;

    vec4 res = raymarchCast(ro, rd);
    vec3 m = res.rgb;
    float t = res.a;

    if ( m.r + m.g + m.b > 0.0 ) 
    {
        vec3 pos = ro + t*rd;
        vec3 nor = raymarchNormal( pos );
        vec3 ref = reflect( rd, nor );
        col = m;
        
        #if defined(RAYMARCH_FLOOR)
        if ( m.y < 0.0) {
            col = RAYMARCH_FLOOR;
        }
        #endif

        // lighitng        
        float occ = raymarchAO( pos, nor );
        vec3  lig = normalize( LIGHT_POSITION );
        vec3  hal = normalize( lig-rd );
        float amb = clamp( 0.5+0.5*nor.y, 0.0, 1.0 );
        float dif = clamp( dot( nor, lig ), 0.0, 1.0 );
        float bac = clamp( dot( nor, normalize(vec3(-lig.x,0.0,-lig.z))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);
        float dom = smoothstep( -0.1, 0.1, ref.y );
        float fre = pow( clamp(1.0+dot(nor,rd),0.0,1.0), 2.0 );
        
        dif *= raymarchSoftShadow( pos, lig, 0.02, 2.5 );
        dom *= raymarchSoftShadow( pos, ref, 0.02, 2.5 );

        vec3 lin = vec3(0.0);
        lin += 1.30 * dif * LIGHT_COLOR;
        lin += 0.40 * amb * occ * RAYMARCH_AMBIENT;
        lin += 0.50 * dom * occ * RAYMARCH_AMBIENT;
        lin += 0.50 * bac * occ * 0.25;
        lin += 0.25 * fre * occ;
        col = col*lin;

        // float spe = pow( clamp( dot( nor, hal ), 0.0, 1.0 ),16.0)*
        //             dif *
        //             (0.04 + 0.96*pow( clamp(1.0+dot(hal,rd),0.0,1.0), 5.0 ));
        // col += 10.00*spe*vec3(1.00,0.90,0.70);

        // col = mix( col, vec3(0.8,0.9,1.0), 1.0-exp( -0.0002*t*t*t ) );

        // col = vec3(1.) * amb;
        // col = vec3(1.) * dif;
        // col = vec3(1.) * bac;
        // col = vec3(1.) * dom;
        // col = vec3(1.) * fre;
        // col = vec3(1.) * spe;
        // col = ref;
        // col = nor;
    }

    return vec4( saturate(col), t );
}

#endif