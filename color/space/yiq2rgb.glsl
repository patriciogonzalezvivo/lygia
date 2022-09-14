/*
original_author: Patricio Gonzalez Vivo
description: pass a color in YIQ and get RGB color. From https://en.wikipedia.org/wiki/YIQ
use: yiq2rgb(<vec3|vec4> color)
*/

#ifndef FNC_YIQ2RGB
#define FNC_YIQ2RGB

const mat3 yiq2rgb_mat = mat3(1.0, 0.956, 0.621, 1.0, -0.272, -0.647, 1.0, -1.105, 1.702);

vec3 yiq2rgb(in vec3 yiq) {
  return yiq2rgb_mat * yiq;
}

vec4 yiq2rgb(in vec4 yiq) {
    return vec4(yiq2rgb(yiq.rgb), yiq.a);
}
#endif
