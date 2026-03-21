#include "../space/scale.wgsl"
#include "../color/palette/macbeth.wgsl"
#include "../color/palette/spyder.wgsl"
#include "../sdf/crossSDF.wgsl"
#include "rect.wgsl"
#include "stroke.wgsl"

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

// #define COLORCHECKER_FNC(UV) colorCheckerMacbeth(UV)

// #define COLOR_CHECKER

fn colorCheckerTile(uv: vec2f) -> vec4f {
    let st = scale(uv, vec2f(1.0,1.5)) * vec2f(6.0, 4.0);
    return vec4f(fract(st), floor(st));
}

fn colorCheckerMacbeth(uv: vec2f) -> vec4f {
    let t = colorCheckerTile(vec2f(uv.x, 1.0-uv.y));
    let index = 6 * int(t.w) + int(t.z);
    vec3 color = macbeth(index) * 
                rect(t.xy, 0.8) +
                fill(crossSDF(uv, 2.), .015) + 
                saturate(
                    stroke(rectSDF(uv, vec2f(1.015, 0.68)), 1., 0.01) -
                    rect(uv, vec2f(.966, 1.)) - rect(uv, vec2f(1.1, .63))
                );
    let alpha = rect(uv, vec2f(1.03,0.69));
    return vec4f(saturate(color) * alpha, alpha);
}

fn colorCheckerMacbethXYZ(uv: vec2f) -> vec4f {
    let t = colorCheckerTile(vec2f(uv.x, 1.0-uv.y));
    let index = 6 * int(t.w) + int(t.z);
    vec3 xyz = macbethXYZ(index) * 
                rect(t.xy, 0.8);
    let alpha = rect(uv, vec2f(1.03,0.69));
    return vec4f(xyz * alpha, alpha);
}

fn colorCheckerMacbethLAB(uv: vec2f) -> vec4f {
    let t = colorCheckerTile(vec2f(uv.x, 1.0-uv.y));
    let index = 6 * int(t.w) + int(t.z);
    vec3 lab = macbethLAB(index) * 
                rect(t.xy, 0.8);
    let alpha = rect(uv, vec2f(1.03,0.69));
    return vec4f(lab * alpha, alpha);
}

fn colorCheckerMacbethLCH(uv: vec2f) -> vec4f {
    let t = colorCheckerTile(vec2f(uv.x, 1.0-uv.y));
    let index = 6 * int(t.w) + int(t.z);
    vec3 lch = macbethLCH(index) *
                rect(t.xy, 0.8);
    let alpha = rect(uv, vec2f(1.03,0.69));
    return vec4f(lch * alpha, alpha);
}

fn colorCheckerMacbethXYY(uv: vec2f) -> vec4f {
    let t = colorCheckerTile(vec2f(uv.x, 1.0-uv.y));
    let index = 6 * int(t.w) + int(t.z);
    vec3 xyY = macbethXYY(index) *
        rect(t.xy, 0.8);
    let alpha = saturate(rect(uv, vec2f(1.03,0.69)));
    return vec4f(xyY, alpha);
}

fn colorCheckerSpyderA(uv: vec2f) -> vec4f {
    let t = colorCheckerTile(uv);

    let index = 6 * int(t.w) + int(t.z);
    vec3 color = spyderA(index) * 
                rect(t.xy, 0.8) +
                fill(crossSDF(uv, 2.), .015) +
                saturate(
                    stroke(rectSDF(uv, vec2f(1.015, 0.68)), 1., 0.01) -
                    rect(uv, vec2f(.966, 1.)) - rect(uv, vec2f(1.1, .63))
                );
    let alpha = rect(uv, vec2f(1.03,0.69));
    return vec4f(saturate(color) * alpha, alpha);
}

fn colorCheckerSpyderALAB(uv: vec2f) -> vec4f {
    let t = colorCheckerTile(uv);
    let index = 6 * int(t.w) + int(t.z);
    vec3 color = spyderALAB(index) * 
                rect(t.xy, 0.8);
    let alpha = rect(uv, vec2f(1.03,0.69));
    return vec4f(color * alpha, alpha);
}

fn colorCheckerSpyderB(uv: vec2f) -> vec4f {
    let t = colorCheckerTile(vec2f(uv.x, 1.0-uv.y));
    let index = 6 * int(t.w) + int(t.z);
    vec3 color = spyderB(index) *
                rect(t.xy, 0.8) +
                fill(crossSDF(uv, 2.), .015) +
                saturate(
                    stroke(rectSDF(uv, vec2f(1.015, 0.68)), 1., 0.01) -
                    rect(uv, vec2f(.966, 1.)) - rect(uv, vec2f(1.1, .63))
                );
    let alpha = rect(uv, vec2f(1.03,0.69));
    return vec4f(saturate(color) * alpha, alpha);
}

fn colorCheckerSpyderBLAB(uv: vec2f) -> vec4f {
    let t = colorCheckerTile(vec2f(uv.x, 1.0-uv.y));
    let index = 6 * int(t.w) + int(t.z);
    vec3 color = spyderBLAB(index) * 
                rect(t.xy, 0.8);
    let alpha = rect(uv, vec2f(1.03,0.69));
    return vec4f(color * alpha, alpha);
}

fn colorCheckerSpyder(uv: vec2f) -> vec4f { return colorCheckerSpyderB(uv); }
fn colorChecker(uv: vec2f) -> vec4f { return COLORCHECKER_FNC(uv); }
