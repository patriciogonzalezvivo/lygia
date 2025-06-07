/*
contributors:
    - Patricio Gonzalez Vivo
    - David Hoskins
    - Inigo Quilez
description: Pass a value and get some random normalize value between 0 and 1
notes:
    - While the GLSL and HLSL versions of this file support a RANDOM_HIGHER_RANGE option, the current implementation has this permanently enabled.
options:
    - RANDOM_SINLESS: Use sin-less random, which tolerates bigger values before producing pattern. From https://www.shadertoy.com/view/4djSRW
    - RANDOM_SCALE: by default this scale if for number with a big range. For producing good random between 0 and 1 use bigger range
examples:
    - /shaders/generative_random.frag
license:
    - MIT License (MIT) Copyright 2014, David Hoskins
*/

const RANDOM_SINLESS: bool = true;

const RANDOM_SCALE: vec4f = vec4f(.1031, .1030, .0973, .1099);

fn random(p: f32) -> f32 {
    var x = p;
    if (RANDOM_SINLESS) {
        x = fract(x * RANDOM_SCALE.x);
        x *= x + 33.33;
        x *= x + x;
        return fract(x);
    } else {
        return fract(sin(x) * 43758.5453);
    }
}

fn random2(st: vec2f) -> f32 {
    if (RANDOM_SINLESS) {
        var p3  = fract(vec3(st.xyx) * RANDOM_SCALE.xyz);
        p3 += dot(p3, p3.yzx + 33.33);
        return fract((p3.x + p3.y) * p3.z);
    } else {
        return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453);
    }
}

fn random3(p: vec3f) -> f32 {
    var pos = p;
    if (RANDOM_SINLESS) {
        pos  = fract(pos * RANDOM_SCALE.xyz);
        pos += dot(pos, pos.zyx + 31.32);
        return fract((pos.x + pos.y) * pos.z);
    } else {
        return fract(sin(dot(pos.xyz, vec3(70.9898, 78.233, 32.4355))) * 43758.5453123);
    }
}

fn random4(p: vec4f) -> f32 {
    var pos = p;
    if (RANDOM_SINLESS) {
        pos = fract(pos * RANDOM_SCALE);
        pos += dot(pos, pos.wzxy+33.33);
        return fract((pos.x + pos.y) * (pos.z + pos.w));
    } else {
        let dot_product = dot(pos, vec4(12.9898,78.233,45.164,94.673));
        return fract(sin(dot_product) * 43758.5453);
    }
}

fn random21(p: f32) -> vec2f {
    var p3 = fract(vec3(p) * RANDOM_SCALE.xyz);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn random22(p: vec2f) -> vec2f {
    var p3 = fract(p.xyx * RANDOM_SCALE.xyz);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn random23(p: vec3f) -> vec2f {
    var p3 = p;
    p3 = fract(p3 * RANDOM_SCALE.xyz);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn random31(p: f32) -> vec3f {
    var p3 = fract(vec3(p) * RANDOM_SCALE.xyz);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.xxy + p3.yzz) * p3.zyx);
}

fn random32(p: vec2f) -> vec3f {
    var p3 = fract(vec3(p.xyx) * RANDOM_SCALE.xyz);
    p3 += dot(p3, p3.yxz + 19.19);
    return fract((p3.xxy + p3.yzz) * p3.zyx);
}

fn random33(p_: vec3f) -> vec3f {
    var p = fract(p_ * RANDOM_SCALE.xyz);
    p += dot(p, p.yxz + 19.19);
    return fract((p.xxy + p.yzz) * p.zyx);
}

fn random41(p: f32) -> vec4f {
    var p4 = fract(p * RANDOM_SCALE);
    p4 += dot(p4, p4.wzxy + 19.19);
    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}

fn random42(p: vec2f) -> vec4f {
    var p4 = fract(p.xyxy * RANDOM_SCALE);
    p4 += dot(p4, p4.wzxy + 19.19);
    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}

fn random43(p: vec3f) -> vec4f {
    var p4 = fract(p.xyzx  * RANDOM_SCALE);
    p4 += dot(p4, p4.wzxy + 19.19);
    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}

fn random44(p: vec4f) -> vec4f {
    var p4 = p;
    p4 = fract(p4  * RANDOM_SCALE);
    p4 += dot(p4, p4.wzxy + 19.19);
    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}
