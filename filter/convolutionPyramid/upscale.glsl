#include "../../math/absi.glsl"

#ifndef CONVOLUTIONPYRAMID_H1
#define CONVOLUTIONPYRAMID_H1 1.0334, 0.6836, 0.1507
#endif

#ifndef CONVOLUTIONPYRAMID_H2
#define CONVOLUTIONPYRAMID_H2 0.0270
#endif

#ifndef CONVOLUTIONPYRAMID_G
#define CONVOLUTIONPYRAMID_G 0.7753, 0.0312
#endif

#ifndef FNC_CONVOLUTIONPYRAMID_UPSCALE
#define FNC_CONVOLUTIONPYRAMID_UPSCALE
vec4 convolutionPyramidUpscale(sampler2D tex0, sampler2D tex1, vec2 st, vec2 pixel) {
    const vec3  h1 = vec3(CONVOLUTIONPYRAMID_H1);
    const float h2 = CONVOLUTIONPYRAMID_H2;
    const vec2  g  = vec2(CONVOLUTIONPYRAMID_G);

    vec4 color = vec4(0.0);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            vec2 uv = st + vec2(float(dx), float(dy)) * pixel;
            color += texture2D(tex0, uv) * g[ absi(dx) ] * g[ absi(dy) ];
        }
    }

    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            vec2 uv = st + vec2(float(dx), float(dy)) * pixel * 2.;
            color += texture2D(tex1, uv) * h2 * h1[ absi(dx) ] * h1[ absi(dy) ];
        }
    }

    return color;
}
#endif