#include "../math/map.wgsl"

/*
contributors: Kathy McGuiness
description: |
    Returns a gear shaped SDF
    Some notes about the parameters:
        * b determines the length and roundness of the spokes
        * n is the number of spokes 
use: 
    - gearSDF(<vec2> st, <vec2> center, <float> b, <int> n_spokes);
    - gearSDF(<vec2> st, <float> b, <int> n_spokes);
options:
    - CENTER_2D: <vec2> the center of the gear
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
    - https://gist.githubusercontent.com/kfahn22/84ded9666e6037fdb62376ccffb6582e/raw/14bd8fb7911b0dc5aec635357693e21fdda512bc/gear.fr
*/

fn gearSDF(st: vec2f, b: f32, N: i32) -> f32 {
    let e = 2.71828182845904523536;
    st -= CENTER_2D;
    st -= 0.5;
    st *= 3.0;
    let s = map(b, 1.0, 15.0, 0.066, 0.5);
    let d = length(st) - s;
    let omega = b * sin(float(N) * atan(st.y, st.x));
    let l = pow(e, 2.0 * omega);
    let hyperTan = (l - 1.0) / (l + 1.0);
    let r = (1.0/b) * hyperTan;
    return (d + min(d, r));
}
