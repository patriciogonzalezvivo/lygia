import lygia::space::sqTile::sqTile;
import lygia::math::mirror::mirror2;

/*
contributors: Patricio Gonzalez Vivo
description: mirror a tiles pattern
use:
    - <vec4> mirrorTile(<vec4> tile)
    - <vec4> mirrorTile(<vec2> st [, <float|vec2> scale])
    - <vec4> mirrorXTile(<vec4> tile)
    - <vec4> mirrorXTile(<vec2> st [, <float|vec2> scale])
    - <vec4> mirrorYTile(<vec4> tile)
    - <vec4> mirrorYTile(<vec2> st [, <float|vec2> scale])
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_tiles.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn mirrorTile(t: vec4f) -> vec4f {
    return vec4f(mirror2(t.xy + t.zw), t.z, t.w);
}

fn mirrorTile2(v: vec2f) -> vec4f {
    return mirrorTile(sqTile(v));
}

fn mirrorTile_s(v: vec2f, s: f32) -> vec4f {
    return mirrorTile2(v * s);
}

fn mirrorTile_s2(v: vec2f, s: vec2f) -> vec4f {
    return mirrorTile2(v * s);
}

fn mirrorXTile(t: vec4f) -> vec4f {
    return vec4f(mirror2(vec2f(t.x + t.z, t.y)).x, t.y, t.z, t.w);
}

fn mirrorXTile2(v: vec2f) -> vec4f {
    return mirrorXTile(sqTile(v));
}

fn mirrorXTile_s(v: vec2f, s: f32) -> vec4f {
    return mirrorXTile2(v * s);
}

fn mirrorXTile_s2(v: vec2f, s: vec2f) -> vec4f {
    return mirrorXTile2(v * s);
}

fn mirrorYTile(t: vec4f) -> vec4f {
    return vec4f(t.x, mirror2(vec2f(t.y + t.w, t.x)).x, t.z, t.w);
}

fn mirrorYTile2(v: vec2f) -> vec4f {
    return mirrorYTile(sqTile(v));
}

fn mirrorYTile_s(v: vec2f, s: f32) -> vec4f {
    return mirrorYTile2(v * s);
}

fn mirrorYTile_s2(v: vec2f, s: vec2f) -> vec4f {
    return mirrorYTile2(v * s);
}
