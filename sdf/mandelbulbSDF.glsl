#include "../space/cart2polar.glsl"

/*
original_author: Kathy McGuiness
description: Returns the mandelbulb SDF
use: mandelbulbSDF(<vec2> st)
*/

#ifndef FNC_MANDELBULBSDF
#define FNC_MANDELBULBSDF
float mandelbulbSDF( in vec3 st ) {
   vec3 zeta = st;
   float m = dot(st,st);
   float dz = 1.0;
   float n = 8.0;
   const int maxiterations = 20;
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
   return 0.25*log(m) * sqrt(m) / dz;
}
#endif