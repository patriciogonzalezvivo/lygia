/*
author: Hugh Kennedy (https://github.com/hughsk)
description: circular easing. From https://github.com/stackgl/glsl-easings
use: circular<In|Out|InOut>(<float> x)
license: |
  This software is released under the MIT license:
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_CIRCULARIN
#define FNC_CIRCULARIN
float circularIn(in float t) {
  return 1.0 - sqrt(1.0 - t * t);
}
#endif

#ifndef FNC_CIRCULAROUT
#define FNC_CIRCULAROUT
float circularOut(in float t) {
  return sqrt((2.0 - t) * t);
}
#endif

#ifndef FNC_CIRCULARINOUT
#define FNC_CIRCULARINOUT
float circularInOut(in float t) {
  return t < 0.5
    ? 0.5 * (1.0 - sqrt(1.0 - 4.0 * t * t))
    : 0.5 * (sqrt((3.0 - 2.0 * t) * (2.0 * t - 1.0)) + 1.0);
}
#endif
