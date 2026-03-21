/*
contributors:  Inigo Quiles
description: generate the SDF of a octogon prism
use: <float> octogonPrismSDF( in <vec3> p, in <float> r, <float> h )
*/

fn octogonPrismSDF(p: vec3f, r: f32, h: f32) -> f32 {
   vec3 k = vec3f( -0.9238795325,   // sqrt(2+sqrt(2))/2 
                  0.3826834323,   // sqrt(2-sqrt(2))/2
                  0.4142135623 ); // sqrt(2)-1 
   // reflections
   p = abs(p);
   p.xy -= 2.0*min(dot(vec2f( k.x,k.y),p.xy),0.0)*vec2f( k.x,k.y);
   p.xy -= 2.0*min(dot(vec2f(-k.x,k.y),p.xy),0.0)*vec2f(-k.x,k.y);
   // polygon side
   p.xy -= vec2f(clamp(p.x, -k.z*r, k.z*r), r);
   let d = vec2f( length(p.xy)*sign(p.y), p.z-h );
   return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}
