#include "../math/saturate.wgsl"

/*
contributors:  Inigo Quiles
description: generate the SDF of s pyramid
use: <float> pyramidSDF(<vec3> p, <float> h )
*/

fn pyramidSDF(p: vec3f, h: f32) -> f32 {
   let m2 = h*h + 0.25;
   
   // symmetry
   p.xz = abs(p.xz);
   p.xz = (p.z>p.x) ? p.zx : p.xz;
   p.xz -= 0.5;

   // project into face plane (2D)
   let q = vec3f( p.z, h*p.y - 0.5*p.x, h*p.x + 0.5*p.y);

   let s = max(-q.x,0.0);
   let t = saturate( (q.y-0.5*p.z)/(m2+0.25) );
   
   let a = m2*(q.x+s)*(q.x+s) + q.y*q.y;
   let b = m2*(q.x+0.5*t)*(q.x+0.5*t) + (q.y-m2*t)*(q.y-m2*t);
   
   let d2 = min(q.y,-q.x*m2-q.y*0.5) > 0.0 ? 0.0 : min(a,b);
   
   // recover 3D and scale, and add sign
   return sqrt( (d2+q.z*q.z)/m2 ) * sign(max(q.z,-p.y));;
}
