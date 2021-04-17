/*
author: Inigo Quiles
description: pass a color in HSB and get RGB color. Also use as reference http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
use: hsv2rgb(<vec3|vec4> color)
license: |
  This software is released under the MIT license:
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_HSV2RGB
#define FNC_HSV2RGB
vec3 hsv2rgb(in vec3 hsb) {
    vec3 rgb = clamp(abs(mod(hsb.x * 6. + vec3(0., 4., 2.), 
                            6.) - 3.) - 1.,
                      0.,
                      1.);
    #ifdef HSV2RGB_SMOOTH
    rgb = rgb*rgb*(3. - 2. * rgb);
    #endif
    return hsb.z * mix(vec3(1.), rgb, hsb.y);
}

vec4 hsv2rgb(in vec4 hsb) {
    return vec4(hsv2rgb(hsb.rgb), hsb.a);
}
#endif