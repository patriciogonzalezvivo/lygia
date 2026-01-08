/*
contributors: nan
description: |
    Converts the input HDR RGB color into one of 16 debug colors that represent
    the pixel's exposure. When the output is cyan, the input color represents
    middle gray (18% exposure). Every exposure stop above or below middle gray
    causes a color shift.

    The relationship between exposures and colors is:

    -5EV  - black
    -4EV  - darkest blue
    -3EV  - darker blue
    -2EV  - dark blue
    -1EV  - blue
     OEV  - cyan
    +1EV  - dark green
    +2EV  - green
    +3EV  - yellow
    +4EV  - yellow-orange
    +5EV  - orange
    +6EV  - bright red
    +7EV  - red
    +8EV  - magenta
    +9EV  - purple
    +10EV - white

use: tonemapDebug(<vec3f|vec4f> x)
*/

fn tonemapDebug3(x: vec3f) -> vec3f {
    // 16 debug colors + 1 duplicated at the end for easy indexing
    var debugColors = array<vec3f, 17>(
        vec3f(0.0, 0.0, 0.0),         // black
        vec3f(0.0, 0.0, 0.1647),      // darkest blue
        vec3f(0.0, 0.0, 0.3647),      // darker blue
        vec3f(0.0, 0.0, 0.6647),      // dark blue
        vec3f(0.0, 0.0, 0.9647),      // blue
        vec3f(0.0, 0.9255, 0.9255),   // cyan
        vec3f(0.0, 0.5647, 0.0),      // dark green
        vec3f(0.0, 0.7843, 0.0),      // green
        vec3f(1.0, 1.0, 0.0),         // yellow
        vec3f(0.90588, 0.75294, 0.0), // yellow-orange
        vec3f(1.0, 0.5647, 0.0),      // orange
        vec3f(1.0, 0.0, 0.0),         // bright red
        vec3f(0.8392, 0.0, 0.0),      // red
        vec3f(1.0, 0.0, 1.0),         // magenta
        vec3f(0.6, 0.3333, 0.7882),   // purple
        vec3f(1.0, 1.0, 1.0),         // white
        vec3f(1.0, 1.0, 1.0)          // white
    );

    // The 5th color in the array (cyan) represents middle gray (18%)
    // Every stop above or below middle gray causes a color shift
    let l = dot(x, vec3f(0.21250175, 0.71537574, 0.07212251));
    var v = log2(l / 0.18);
    v = clamp(v + 5.0, 0.0, 15.0);
    let index = i32(v);
    return mix(debugColors[index], debugColors[index + 1], v - f32(index));
}

fn tonemapDebug4(x: vec4f) -> vec4f {
    return vec4f(tonemapDebug3(x.rgb), x.a);
}
