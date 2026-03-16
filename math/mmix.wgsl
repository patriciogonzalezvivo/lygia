/*
contributors: Patricio Gonzalez Vivo
description: expands mix to linearly mix more than two values
use: <float|vec2|vec3|vec4> mmix(<float|vec2|vec3|vec4> a, <float|vec2|vec3|vec4> b, <float|vec2|vec3|vec4> c [, <float|vec2|vec3|vec4> d], <float> pct)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn mmix(a: f32, b: f32, c: f32) -> f32 { return mix(a, b, c); }
fn mmix2(a: vec2f, b: vec2f, c: f32) -> vec2f { return mix(a, b, c); }
fn mmix2a(a: vec2f, b: vec2f, c: vec2f) -> vec2f { return mix(a, b, c); }
fn mmix3(a: vec3f, b: vec3f, c: f32) -> vec3f { return mix(a, b, c); }
fn mmix3a(a: vec3f, b: vec3f, c: vec3f) -> vec3f { return mix(a, b, c); }
fn mmix4(a: vec4f, b: vec4f, c: f32) -> vec4f { return mix(a, b, c); }
fn mmix4a(a: vec4f, b: vec4f, c: vec4f) -> vec4f { return mix(a, b, c); }

fn mmixa(a: f32, b: f32, c: f32, pct: f32) -> f32 {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

fn mmix2b(a: vec2f, b: vec2f, c: vec2f, pct: f32) -> vec2f {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

fn mmix2c(a: vec2f, b: vec2f, c: vec2f, pct: vec2f) -> vec2f {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

fn mmix3b(a: vec3f, b: vec3f, c: vec3f, pct: f32) -> vec3f {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

fn mmix3c(a: vec3f, b: vec3f, c: vec3f, pct: vec3f) -> vec3f {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

fn mmix4b(a: vec4f, b: vec4f, c: vec4f, pct: f32) -> vec4f {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

fn mmix4c(a: vec4f, b: vec4f, c: vec4f, pct: vec4f) -> vec4f {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

fn mmixb(a: f32, b: f32, c: f32, d: f32, pct: f32) -> f32 {
    return mix(
        mix(a, b, 3. * pct),
        mix(b,
            mix( c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

fn mmix2d(a: vec2f, b: vec2f, c: vec2f, d: vec2f, pct: f32) -> vec2f {
    return mix(
        mix(a, b, 3. * pct),
        mix(b,
            mix( c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

fn mmix2e(a: vec2f, b: vec2f, c: vec2f, d: vec2f, pct: vec2f) -> vec2f {
    return mix(
        mix(a, b, 3. * pct),
        mix(b,
            mix( c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

fn mmix3d(a: vec3f, b: vec3f, c: vec3f, d: vec3f, pct: f32) -> vec3f {
    return mix(
        mix(a, b, 3. * pct),
        mix(b,
            mix( c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

fn mmix3e(a: vec3f, b: vec3f, c: vec3f, d: vec3f, pct: vec3f) -> vec3f {
    return mix(
        mix(a, b, 3. * pct),
        mix(b,
            mix( c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

fn mmix4d(a: vec4f, b: vec4f, c: vec4f, d: vec4f, pct: f32) -> vec4f {
    return mix(
        mix(a, b, 3. * pct),
        mix(b,
            mix( c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}

fn mmix4e(a: vec4f, b: vec4f, c: vec4f, d: vec4f, pct: vec4f) -> vec4f {
    return mix(
        mix(a, b, 3. * pct),
        mix(b,
            mix( c,
                d,
                3. * (max(pct, .66) - .66)),
            3. * (clamp(pct, .33, .66) - .33)
        ),
        step(.33, pct)
    );
}
