#include "../space/scale.glsl"
#include "../color/palette/macbeth.glsl"
#include "../color/palette/spyder.glsl"
#include "../sdf/crossSDF.glsl"
#include "rect.glsl"
#include "stroke.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Draw a color checker (Macbeth or Spyder)
use:
    - colorChecker(<vec2> uv)
    - colorCheckerMacbeth(<vec2> uv)
    - colorCheckerSpyder(<vec2> uv)
options:
    - COLORCHECKER_FNC: function to use to draw the color checker
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_colorChecker.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef COLORCHECKER_FNC
#define COLORCHECKER_FNC(UV) colorCheckerMacbeth(UV)
#endif

#ifndef COLOR_CHECKER
#define COLOR_CHECKER

vec4 colorCheckerTile(vec2 uv) {
    vec2 st = scale(uv, vec2(1.0,1.5)) * vec2(6.0, 4.0);
    return vec4(fract(st), floor(st));
}

vec4 colorCheckerMacbeth (vec2 uv) {
    vec4 t = colorCheckerTile(vec2(uv.x, 1.0-uv.y));
    int index = 6 * int(t.w) + int(t.z);
    vec3 color = macbeth(index) * 
                rect(t.xy, 0.8) +
                fill(crossSDF(uv, 2.), .015) + 
                saturate(
                    stroke(rectSDF(uv, vec2(1.015, 0.68)), 1., 0.01) -
                    rect(uv, vec2(.966, 1.)) - rect(uv, vec2(1.1, .63))
                );
    float alpha = rect(uv, vec2(1.03,0.69));
    return vec4(saturate(color) * alpha, alpha);
}

vec4 colorCheckerMacbethXYZ(vec2 uv) {
    vec4 t = colorCheckerTile(vec2(uv.x, 1.0-uv.y));
    int index = 6 * int(t.w) + int(t.z);
    vec3 xyz = macbethXYZ(index) * 
                rect(t.xy, 0.8);
    float alpha = rect(uv, vec2(1.03,0.69));
    return vec4(xyz * alpha, alpha);
}

vec4 colorCheckerMacbethLAB(vec2 uv) {
    vec4 t = colorCheckerTile(vec2(uv.x, 1.0-uv.y));
    int index = 6 * int(t.w) + int(t.z);
    vec3 lab = macbethLAB(index) * 
                rect(t.xy, 0.8);
    float alpha = rect(uv, vec2(1.03,0.69));
    return vec4(lab * alpha, alpha);
}

vec4 colorCheckerMacbethLCH(vec2 uv) {
    vec4 t = colorCheckerTile(vec2(uv.x, 1.0-uv.y));
    int index = 6 * int(t.w) + int(t.z);
    vec3 lch = macbethLCH(index) *
                rect(t.xy, 0.8);
    float alpha = rect(uv, vec2(1.03,0.69));
    return vec4(lch * alpha, alpha);
}

vec4 colorCheckerMacbethXYY(vec2 uv) {
    vec4 t = colorCheckerTile(vec2(uv.x, 1.0-uv.y));
    int index = 6 * int(t.w) + int(t.z);
    vec3 xyY = macbethXYY(index) *
        rect(t.xy, 0.8);
    float alpha = saturate(rect(uv, vec2(1.03,0.69)));
    return vec4(xyY, alpha);
}

vec4 colorCheckerSpyderA(vec2 uv) {
    vec4 t = colorCheckerTile(uv);

    int index = 6 * int(t.w) + int(t.z);
    vec3 color = spyderA(index) * 
                rect(t.xy, 0.8) +
                fill(crossSDF(uv, 2.), .015) +
                saturate(
                    stroke(rectSDF(uv, vec2(1.015, 0.68)), 1., 0.01) -
                    rect(uv, vec2(.966, 1.)) - rect(uv, vec2(1.1, .63))
                );
    float alpha = rect(uv, vec2(1.03,0.69));
    return vec4(saturate(color) * alpha, alpha);
}

vec4 colorCheckerSpyderALAB(vec2 uv) {
    vec4 t = colorCheckerTile(uv);
    int index = 6 * int(t.w) + int(t.z);
    vec3 color = spyderALAB(index) * 
                rect(t.xy, 0.8);
    float alpha = rect(uv, vec2(1.03,0.69));
    return vec4(color * alpha, alpha);
}

vec4 colorCheckerSpyderB(vec2 uv) {
    vec4 t = colorCheckerTile(vec2(uv.x, 1.0-uv.y));
    int index = 6 * int(t.w) + int(t.z);
    vec3 color = spyderB(index) *
                rect(t.xy, 0.8) +
                fill(crossSDF(uv, 2.), .015) +
                saturate(
                    stroke(rectSDF(uv, vec2(1.015, 0.68)), 1., 0.01) -
                    rect(uv, vec2(.966, 1.)) - rect(uv, vec2(1.1, .63))
                );
    float alpha = rect(uv, vec2(1.03,0.69));
    return vec4(saturate(color) * alpha, alpha);
}

vec4 colorCheckerSpyderBLAB(vec2 uv) {
    vec4 t = colorCheckerTile(vec2(uv.x, 1.0-uv.y));
    int index = 6 * int(t.w) + int(t.z);
    vec3 color = spyderBLAB(index) * 
                rect(t.xy, 0.8);
    float alpha = rect(uv, vec2(1.03,0.69));
    return vec4(color * alpha, alpha);
}

vec4 colorCheckerSpyder(vec2 uv) { return colorCheckerSpyderB(uv); }
vec4 colorChecker (vec2 uv){ return COLORCHECKER_FNC(uv); }

#endif