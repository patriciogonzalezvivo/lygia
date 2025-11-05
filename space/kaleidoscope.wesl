import lygia::math::consts::TWO_PI;

/*
contributors: [Mario Carrillo, Daniel Ilett]
description: |
    Converts carteesian coordinates into polar coordinates
    emulating a kaleidoscope visual effect.
    Based on Daniel Ilett's tutorial on reflecting polar coordinates:
    https://danielilett.com/2020-02-19-tut3-8-crazy-kaleidoscopes/
*/

// Use when when you want to specify the segment count and the phase for animation
fn kaleidoscope_full(coord: vec2f, segmentCount: f32, phase: f32) -> vec2f {
    var uv: vec2f;
    uv = coord - 0.5;

    let radius = length(uv);
    var angle = atan2(uv.y, uv.x);

    let segmentAngle = TWO_PI / segmentCount;
    angle -= segmentAngle * floor(angle / segmentAngle);
    angle = min(angle, segmentAngle - angle);

    var kuv = vec2f(cos(angle + phase), sin(angle + phase)) * radius + 0.5;
    kuv = max(min(kuv, 2.0 - kuv), -kuv);

    return kuv;
}

// Default use when just the coordinates are given, the segment count is set to 8
fn kaleidoscope(coord: vec2f) -> vec2f {
    return kaleidoscope_full(coord, 8.0, 0.0);
}

// Use when when you want to specify the segment count
fn kaleidoscope_seg(coord: vec2f, segmentCount: f32) -> vec2f {
    return kaleidoscope_full(coord, segmentCount, 0.0);
}
