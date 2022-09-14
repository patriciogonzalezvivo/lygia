/*
original_author: Patricio Gonzalez Vivo
description: Convert from RGB to YIQ which was the followin range
use: rgb2yiq(<vec3|vec4> color)
*/

#ifndef FNC_RGB2YIQ
#define FNC_RGB2YIQ
const mat3 rgb2yiq_mat = mat3(0.299, 0.587, 0.114, 0.596, -0.274, -0.322, 0.212, -0.523, 0.311);

vec3 rgb2yiq(in vec3 rgb) {
  return rgb2yiq_mat * rgb;
}

vec4 rgb2yiq(in vec4 rgb) {
    return vec4(rgb2yiq(rgb.rgb), rgb.a);
}
#endif
