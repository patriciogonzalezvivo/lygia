const MIN_N_DOT_V: f32 = 1e-4;
// Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886"
fn clampNoV(NoV: f32) -> f32 {
    return max(NoV, MIN_N_DOT_V);
}
