/*
contributors: Sam Hocevar
description: Pass a color in RGB and get HSB color. From http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
*/

// LATER make this a @param const when WESL supports that
const HCV_EPSILON: f32 = 1.0e-10;

fn rgb2hsv(c: vec3f) -> vec3f {
    let K = vec4f(0.0, -0.33333333333333333333, 0.6666666666666666666, -1.0);
    let p = mix(vec4f(c.bg, K.wz), vec4f(c.gb, K.xy), step(c.b, c.g));
    let q = mix(vec4f(p.xyw, c.r), vec4f(c.r, p.yzx), step(p.x, c.r));
    let d = q.x - min(q.w, q.y);
    return vec3f(abs(q.z + (q.w - q.y) / (6.0 * d + HCV_EPSILON)), d / (q.x + HCV_EPSILON), q.x);
}
fn rgb2hsv4(color: vec4f) -> vec4f {
    return vec4f(rgb2hsv(color.rgb), color.a);
}
