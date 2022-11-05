
/*
original_author: Johan Ismael
description: |
  Color gamma correction similar to Levels adjusment in Photoshop
  Adapted from Romain Dura (http://mouaif.wordpress.com/?p=94)
use: levelsGamma(<vec3|vec4> color, <float|vec3> gamma)
*/

#ifndef FNC_GAMMA
#define FNC_GAMMA
vec3 levelsGamma(in vec3 color, in vec3 gammaAmount) {
  return pow(color, 1. / gammaAmount);
}

vec3 levelsGamma(in vec3 color, in float gammaAmount) {
  return levelsGamma(color, vec3(gammaAmount));
}

vec4 levelsGamma(in vec4 color, in vec3 gammaAmount) {
  return vec4(levelsGamma(color.rgb, gammaAmount), color.a);
}

vec4 levelsGamma(in vec4 color, in float gammaAmount) {
  return vec4(levelsGamma(color.rgb, gammaAmount), color.a);
}
#endif