/*
original_author: Patricio Gonzalez Vivo
description: |
  Convert from RGB to YIQ which was the followin range
  Y [0,.1], I [-0.5957, 0.5957], Q [-0.5226, 0.5226]
  From https://en.wikipedia.org/wiki/YIQ
use: rgb2yiq(<float3|float4> color)
*/

#ifndef FNC_RGB2YIQ
#define FNC_RGB2YIQ
// https://en.wikipedia.org/wiki/YIQ
const float3x3 rgb2yiq_mat = float3x3(
    .299,  .596,  .211,
    .587, -.274, -.523,
    .114, -.322,  .0312
);

float3 rgb2yiq(in float3 rgb) {
  return mul(rgb2yiq_mat, rgb);
}

float4 rgb2yiq(in float4 rgb) {
    return float4(rgb2yiq(rgb.rgb), rgb.a);
}
#endif
