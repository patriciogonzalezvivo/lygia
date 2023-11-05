#include "../space/cart2polar.glsl"

/*
contributors: Kathy McGuiness
description: |
    Returns the mandelbulb SDF
    For more information about the mandlebulb, check [this article](https://en.wikipedia.org/wiki/Mandelbulb)

use: mandelbulbSDF(<vec2> st)
examples:
    - https://gist.githubusercontent.com/kfahn22/4c29e6d2bdc33d639f315edaa934d287/raw/0eb94683614995476fb63cfa99881e420f1c5be6/mandelbulb.frag
*/

#ifndef FNC_MANDELBULBSDF
#define FNC_MANDELBULBSDF
vec2 mandelbulbSDF( in vec3 st ) {
   vec3 zeta = st;
   float m = dot(st,st);
   float dz = 1.0;
   float n = 8.0;
   const int maxiterations = 20;
   float iterations = 0.0;
   float r = 0.0; 
   float dr = 1.0;
   for (int i = 0; i < maxiterations; i+=1) {
       dz = n*pow(m, 3.5)*dz + 1.0;
       vec3 sphericalZ = cart2polar( zeta ); 
       float newx = pow(sphericalZ.x, n) * sin(sphericalZ.y*n) * cos(sphericalZ.z*n);
       float newy = pow(sphericalZ.x, n) * sin(sphericalZ.y*n) * sin(sphericalZ.z*n);
       float newz = pow(sphericalZ.x, n) * cos(sphericalZ.y*n);
       zeta.x = newx + st.x;
       zeta.y = newy + st.y;
       zeta.z = newz + st.z;

       m = dot(zeta, zeta);
       if ( m > 2.0 )
         break;
   }
 
   // distance estimation through the Hubbard-Douady potential from Inigo Quilez
   return vec2(0.25*log(m) * sqrt(m) / dz, iterations);
}
#endif