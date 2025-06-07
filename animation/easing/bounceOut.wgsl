/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Bounce out easing. From https://github.com/stackgl/glsl-easings
use: <f32> bounceOut(<f32> x)
examples:
    - https://raw.githubusercontent.com/eduardfossas/lygia-study-examples/main/animation/e_EasingBounce.frag
*/

fn bounceOut(t: f32) -> f32 {
    let a = 4.0 / 11.0;
    let b = 8.0 / 11.0;
    let c = 9.0 / 10.0;

    let ca = 4356.0 / 361.0;
    let cb = 35442.0 / 1805.0;
    let cc = 16061.0 / 1805.0;

    let t2 = t * t;

    return select(
      select(
        select(
          10.8 * t * t - 20.52 * t + 10.72,
          ca * t2 - cb * t + cc,
          t < c
        ),
        9.075 * t2 - 9.9 * t + 3.4,
        t < b
      ),
      7.5625 * t2,
      t < a
    );
}
