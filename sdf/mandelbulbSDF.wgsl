#include "../space/cart2polar.wgsl"

/*
contributors: Kathy McGuiness
description: |
    Returns the mandelbulb SDF
    For more information about the mandlebulb, check [this article](https://en.wikipedia.org/wiki/Mandelbulb)

use: mandelbulbSDF(<vec2> st)
examples:
    - https://gist.githubusercontent.com/kfahn22/4c29e6d2bdc33d639f315edaa934d287/raw/0eb94683614995476fb63cfa99881e420f1c5be6/mandelbulb.frag
*/

fn mandelbulbSDF(st: vec3f) -> vec2f {
   let zeta = st;
   let m = dot(st,st);
   let dz = 1.0;
   let n = 8.0;
   let maxiterations = 20;
   let iterations = 0.0;
   let r = 0.0;
   let dr = 1.0;
   for (int i = 0; i < maxiterations; i+=1) {
       dz = n*pow(m, 3.5)*dz + 1.0;
       let sphericalZ = cart2polar( zeta );
       let newx = pow(sphericalZ.x, n) * sin(sphericalZ.y*n) * cos(sphericalZ.z*n);
       let newy = pow(sphericalZ.x, n) * sin(sphericalZ.y*n) * sin(sphericalZ.z*n);
       let newz = pow(sphericalZ.x, n) * cos(sphericalZ.y*n);
       zeta.x = newx + st.x;
       zeta.y = newy + st.y;
       zeta.z = newz + st.z;

       m = dot(zeta, zeta);
       if ( m > 2.0 )
         break;
   }
 
   // distance estimation through the Hubbard-Douady potential from Inigo Quilez
   return vec2f(0.25*log(m) * sqrt(m) / dz, iterations);
}
