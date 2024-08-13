/*
contributors: Inigo Quiles
description: Add fog to the scene. See https://iquilezles.org/articles/fog/
use: vec3 raymarchFog(in <vec3> pixelColor, float distanceToPoint)
options:
    - FOG_DENSITY
    - FOG_FALLOFF
    - FOG_COLOR_COOL
    - FOG_COLOR_WARM
*/

#ifndef FOG_DENSITY
#define FOG_DENSITY 0.0
#endif

#ifndef FOG_FALLOFF
#define FOG_FALLOFF 0.0
#endif

#ifndef FOG_COLOR_COOL
#define FOG_COLOR_COOL vec3(0.5, 0.6, 0.7)
#endif

#ifndef FOG_COLOR_WARM
#define FOG_COLOR_WARM vec3(1.0, 0.9, 0.7)
#endif

#ifndef FNC_RAYMARCH_FOG
#define FNC_RAYMARCH_FOG

vec3 raymarchFog(in vec3 col,   // color of pixel
                in float t) {   // distance to point
    float fogAmount = 1.0 - exp(-t * FOG_DENSITY);
    return mix(col, FOG_COLOR_COOL, fogAmount);
}

vec3 raymarchColorFog(in vec3 col,      // color of pixel
                      in float t,       // distance to point
                      in vec3 rd,       // camera to point
                      in vec3 lig) {    // sun direction
    float fogAmount = 1.0 - exp(-t * FOG_DENSITY);
    float sunAmount = max(dot(rd, lig), 0.0);
    vec3 fogColor = mix(FOG_COLOR_COOL, FOG_COLOR_WARM, pow(sunAmount, 8.0));
    return mix(col, fogColor, fogAmount);
}

vec3 raymarchHeightFog( in vec3 col,     // color of pixel
                        in float t,      // distnace to point
                        in vec3 ro,      // camera position
                        in vec3 rd) {    // camera to point vector
    float fogAmount = (FOG_DENSITY / FOG_FALLOFF) * exp(-ro.y * FOG_FALLOFF) * (1.0 - exp(-t * rd.y * FOG_FALLOFF)) / rd.y;
    return mix(col, FOG_COLOR_COOL, saturate(fogAmount));
}

vec3 raymarchFog(in vec3 col, in float t, in vec3 ro, in vec3 rd){
    if (FOG_DENSITY > 0.0 && FOG_FALLOFF > 0.0) {
        return raymarchHeightFog(col, t, ro, rd);
    }
    else if (FOG_DENSITY > 0.0) {
        #ifdef LIGHT_DIRECTION
            return raymarchColorFog(col, t, rd, LIGHT_DIRECTION);
        #else
            return raymarchFog(col, t);
        #endif
    }
    else {
        return col;
    }
}

#endif
