/*
original_author: Patricio Gonzalez Vivo
description: generate a matrix to change a the saturation of any color
use: saturationMatrix(<float> amount)
*/

#ifndef FNC_SATURATIONMATRIX
#define FNC_SATURATIONMATRIX
mat4 saturationMatrix(in float amount) {
    vec3 lum = vec3(.3086, .6094, .0820 );

    float invAmount= 1. - amount;

    vec3 red = vec3(lum.x * invAmount);
    red += vec3(amount, .0, .0);

    vec3 green = vec3(lum.y * invAmount);
    green += vec3( .0, amount, .0);

    vec3 blue = vec3(lum.z * invAmount);
    blue += vec3( .0, .0, amount);

    return mat4(red,        .0,
                green,      .0,
                blue,       .0,
                .0, .0, .0, 1.);
}
#endif
