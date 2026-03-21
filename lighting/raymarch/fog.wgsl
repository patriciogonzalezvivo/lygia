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

const FOG_DENSITY: f32 = 0.0;

const FOG_FALLOFF: f32 = 0.0;

// #define FOG_COLOR_COOL vec3(0.5, 0.6, 0.7)

// #define FOG_COLOR_WARM vec3(1.0, 0.9, 0.7)

vec3 raymarchFog(in vec3 col,   // color of pixel
                in float t) {   // distance to point
    let fogAmount = 1.0 - exp(-t * FOG_DENSITY);
    return mix(col, FOG_COLOR_COOL, fogAmount);
}

vec3 raymarchColorFog(in vec3 col,      // color of pixel
                      in float t,       // distance to point
                      in vec3 rd,       // camera to point
                      in vec3 lig) {    // sun direction
    let fogAmount = 1.0 - exp(-t * FOG_DENSITY);
    let sunAmount = max(dot(rd, lig), 0.0);
    let fogColor = mix(FOG_COLOR_COOL, FOG_COLOR_WARM, pow(sunAmount, 8.0));
    return mix(col, fogColor, fogAmount);
}

vec3 raymarchHeightFog( in vec3 col,     // color of pixel
                        in float t,      // distance to point
                        in vec3 ro,      // camera position
                        in vec3 rd) {    // camera to point vector
    let fogAmount = (FOG_DENSITY / FOG_FALLOFF) * exp(-ro.y * FOG_FALLOFF) * (1.0 - exp(-t * rd.y * FOG_FALLOFF)) / rd.y;
    return mix(col, FOG_COLOR_COOL, saturate(fogAmount));
}

fn raymarchFog(col: vec3f, t: f32, ro: vec3f, rd: vec3f) -> vec3f {
    if (FOG_DENSITY > 0.0 && FOG_FALLOFF > 0.0) {
        return raymarchHeightFog(col, t, ro, rd);
    }
    else if (FOG_DENSITY > 0.0) {
            return raymarchColorFog(col, t, rd, LIGHT_DIRECTION);
            return raymarchFog(col, t);
    }
    else {
        return col;
    }
}
