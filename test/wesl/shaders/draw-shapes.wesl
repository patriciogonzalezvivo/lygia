// WESL version of lygia_examples/draw_shapes.frag
// Displays 14 different 2D SDF shapes in a 4×4 grid for visual comparison with GLSL

import lygia::draw::fill::fill;
import lygia::space::ratio::ratio;
import lygia::space::rotate::rotate;

import lygia::sdf::circleSDF::circleSDF;
import lygia::sdf::crossSDF::crossSDF;
import lygia::sdf::flowerSDF::flowerSDF;
import lygia::sdf::gearSDF::gearSDF;
import lygia::sdf::heartSDF::heartSDF;
import lygia::sdf::hexSDF::hexSDF;
import lygia::sdf::polySDF::polySDF;
import lygia::sdf::rectSDF::rectSDF;
import lygia::sdf::raysSDF::raysSDF;
import lygia::sdf::rhombSDF::rhombSDF;
import lygia::sdf::spiralSDF::spiralSDF;
import lygia::sdf::starSDF::starSDF1;
import lygia::sdf::triSDF::triSDF;
import lygia::sdf::vesicaSDF::vesicaSDF;

@group(0) @binding(0) var<uniform> uniforms: test::Uniforms;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    var color = vec4f(0.0, 0.0, 0.0, 1.0);
    let pixel = 1.0 / uniforms.resolution;
    var st = pos.xy * pixel;

    // Apply aspect ratio correction
    st = ratio(st, uniforms.resolution);

    // Create 4×4 grid
    let cols = 4.0;
    st *= cols;
    let st_i = floor(st);
    let st_f = fract(st);

    // Rotate each cell by time (matches GLSL u_time * 0.4)
    // Negate rotation to maintain clockwise direction: WGSL's Y-down coordinates
    // reverse the visual direction of rotation compared to GLSL's Y-up coordinates
    let st_f_rotated = rotate(st_f, -uniforms.time * 0.4);

    // Calculate flat index for shape selection (0-15)
    // WGSL @builtin(position) has Y=0 at top, so grid indexing doesn't need
    // the Y-flip that GLSL uses (GLSL: (cols - st_i.y - 1.0) vs WGSL: st_i.y)
    let index = st_i.x + st_i.y * cols;

    var sdf = 0.0;

    // Select SDF based on cell index
    if (index < 0.5) {
        sdf = circleSDF(st_f_rotated);
    } else if (index < 1.5) {
        sdf = vesicaSDF(st_f_rotated, 0.25);
    } else if (index < 2.5) {
        sdf = rhombSDF(st_f_rotated);
    } else if (index < 3.5) {
        sdf = triSDF(st_f_rotated);
    } else if (index < 4.5) {
        sdf = rectSDF(st_f_rotated, vec2f(1.0));
    } else if (index < 5.5) {
        sdf = polySDF(st_f_rotated, 5);
    } else if (index < 6.5) {
        sdf = hexSDF(st_f_rotated);
    } else if (index < 7.5) {
        sdf = starSDF1(st_f_rotated, 5);
    } else if (index < 8.5) {
        sdf = flowerSDF(st_f_rotated, 5);
    } else if (index < 9.5) {
        sdf = crossSDF(st_f_rotated, 1.0);
    } else if (index < 10.5) {
        sdf = gearSDF(st_f_rotated, 10.0, 10);
    } else if (index < 11.5) {
        sdf = heartSDF(st_f_rotated);
    } else if (index < 12.5) {
        sdf = raysSDF(st_f_rotated, 14);
    } else if (index < 13.5) {
        sdf = spiralSDF(st_f_rotated, 0.1);
    } else {
        sdf = 1.0;
    }

    // Fill the SDF
    color = vec4f(vec3f(fill(sdf, 0.5)), 1.0);

    return color;
}
