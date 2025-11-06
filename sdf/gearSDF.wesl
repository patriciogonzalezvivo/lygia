@if(CENTER_2D)
import constants::CENTER_2D;
import lygia::math::map::map;

/*
contributors: Kathy McGuiness
description: |
    Returns a gear-shaped signed distance field (SDF)

    Parameters:
    - b: determines the length and roundness of the spokes
    - N: the number of spokes

    WESL conditionals:
    - @if(CENTER_2D): Uses custom 2D center point instead of default center (0.5, 0.5)
use: |
    gearSDF(st: vec2f, b: f32, N: i32) -> f32
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
    - https://gist.githubusercontent.com/kfahn22/84ded9666e6037fdb62376ccffb6582e/raw/14bd8fb7911b0dc5aec635357693e21fdda512bc/gear.fr
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn gearSDF(st_in: vec2f, b: f32, N: i32) -> f32 {
    const e = 2.71828182845904523536;
    var st: vec2f;
    @if(CENTER_2D)
    st = st_in - constants::CENTER_2D;
    @else
    st = st_in - 0.5;

    st *= 3.0;
    let s = map(b, 1.0, 15.0, 0.066, 0.5);
    let d = length(st) - s;
    // Negate Y in atan2 to compensate for WGSL's Y-down coordinates (maintains GLSL tooth pattern)
    let omega = b * sin(f32(N) * atan2(-st.y, st.x));
    let l = pow(e, 2.0 * omega);
    let hyperTan = (l - 1.0) / (l + 1.0);
    let r = (1.0 / b) * hyperTan;
    return d + min(d, r);
}
