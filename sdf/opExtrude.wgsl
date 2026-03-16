/*
contributors:  Inigo Quiles
description: extrude operation of a 2D SDFs into a 3D one
use: <float> opExtrude( in <vec3> p, in <float> sdf, in <float> h )
*/

fn opExtrude(p: vec3f, sdf: f32, h: f32) -> f32 {
    let w = vec2f( sdf, abs(p.z) - h );
  	return min(max(w.x,w.y),0.0) + length(max(w,0.0));
}
