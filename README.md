<img src="https://lygia.xyz/imgs/lygia.svg" width="200" style="display: block; margin-left: auto; margin-right: auto; filter: drop-shadow(2px 3px 4px gray);">

# LYGIA Shader Library [![](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/patriciogonzalezvivo)

Tired of searching, porting and/or reimplementing the same functions over and over? LYGIA is a shader library of reusable functions that can be include easily on your projects. LYGIA is very granular, designed for reusability, performance and flexibility. And can be easily be added to any projects and frameworks.

<p style="text-align: center;" >
    <a href="https://github.com/patriciogonzalezvivo/lygia_unity_examples"><img src="https://lygia.xyz/imgs/unity.png" width="100" /></a>
    <a href="https://github.com/guidoschmidt/lygia_threejs_examples"><img src="https://lygia.xyz/imgs/threejs.png" width="100" /></a>
    <a href="https://github.com/patriciogonzalezvivo/lygia_p5_examples"><img src="https://lygia.xyz/imgs/p5.png" width="100" /></a>
    <a href="https://editor.p5js.org/patriciogonzalezvivo/sketches"><img src="https://lygia.xyz/imgs/p5js.png" width="100" /></a>
    <a href="https://github.com/patriciogonzalezvivo/lygia_of_examples"><img src="https://lygia.xyz/imgs/of.png" width="100" /></a>
    <a href="https://github.com/vectorsize/lygia-td"><img title="static-resolver by vectorsize" src="https://lygia.xyz/imgs/td.png" width="100" /></a>
    <a href="https://github.com/patriciogonzalezvivo/lygia_examples"><img src="https://lygia.xyz/imgs/glslViewer.png" width="100" /></a>
    <a href="https://observablehq.com/@radames/hello-lygia-shader-library"><img src="https://lygia.xyz/imgs/ob.png" width="100" /></a>
    <a href="https://codesandbox.io/s/lygia-react-starter-fftx6p"><img src="https://lygia.xyz/imgs/r3f.png" width="100" /></a>
    <a href="https://www.figma.com/community/plugin/1138854718618193875"><img src="https://lygia.xyz/imgs/figma.png" width="100" /></a>
    <a href="https://github.com/irmf/irmf-examples/tree/master/examples/028-lygia"><img src="https://lygia.xyz/imgs/irmf.png" width="100" /></a>
    <a href="https://github.com/ossia/score-examples"><img src="https://lygia.xyz/imgs/ossia.png" width="100" /></a>
    <a href="https://www.npmjs.com/package/resolve-lygia"><img src="https://lygia.xyz/imgs/npm.png" width="100" /></a>

</p>

## How to use it?

In your shader `#include` the functions you need:

<div class="codeAndCanvas" data="example.frag">

	#ifdef GL_ES
    precision mediump float;
    #endif

    uniform vec2    u_resolution;
    uniform float   u_time;

    #include "lygia/space/ratio.glsl"
    #include "lygia/math/decimate.glsl"
    #include "lygia/draw/circle.glsl"

    void main(void) {
        vec3 color = vec3(0.0);
        vec2 st = gl_FragCoord.xy/u_resolution.xy;
        st = ratio(st, u_resolution);
        
        color = vec3(st.x,st.y,abs(sin(u_time)));
        color = decimate(color, 20.);
        color += circle(st, .5, .1);
        
        gl_FragColor = vec4(color, 1.0);
    }
    
</div>


### LYGIA Locally

If you are working **locally** on an ecosystem that can resolve `#include` dependencies, just clone LYGIA in your project relative to the shader you are loading:

```bash
    git clone https://github.com/patriciogonzalezvivo/lygia.git
```

or as a submodule:

```bash
    git submodule add https://github.com/patriciogonzalezvivo/lygia.git
```

### LYGIA on the cloud

If you are working on a **cloud platform** probably you want to resolve the dependencies without needing to install anything. Just add a link to `https://lygia.xyz/resolve.js` (JS) or `https://lygia.xyz/resolve.esm.js` (ES6 module): 

```html
    <!-- as a JavaScript source -->
    <script src="https://lygia.xyz/resolve.js"></script>

    <!-- Or as a ES6 module -->
    <script type="module">
        import resolveLygia from "https://lygia.xyz/resolve.esm.js"
    </script>
```

To then resolve the dependencies by passing a `string` or `strings[]` to `resolveLygia()` or `resolveLygiaAsync()`:

```js
    // 1. FIRST

    // Sync resolver, one include at a time
    vertSource = resolveLygia(vertSource);
    fragSource = resolveLygia(fragSource);

    // OR.
    
    // ASync resolver, all includes in parallel calls
    vertSource = resolveLygiaAsync(vertSource);
    fragSource = resolveLygiaAsync(fragSource);
    
    // 2. SECOND

    // Use the resolved source code 
    shdr = createShader(vertSource, fragSource);
```

### Integrations examples

Learn more about how to use it from this **examples**:

* [2D examples for Processing (GLSL)](https://github.com/patriciogonzalezvivo/lygia_p5_examples)
* [2D/3D examples for P5.js (GLSL)](https://editor.p5js.org/patriciogonzalezvivo/sketches)
* [2D examples for Three.js + React (GLSL)](https://codesandbox.io/s/lygia-react-starter-fftx6p) by [Eduard Fossas](https://eduardfossas.vercel.app/)
* [2D examples for Three.js (GLSL)](https://github.com/patriciogonzalezvivo/lygia_threejs_examples)
* [3D examples for Three.js (GLSL)](https://github.com/guidoschmidt/lygia_threejs_examples) by [Guido Schmidt](https://guidoschmidt.cc/)
* [2D examples for OpenFrameworks (GLSL)](https://github.com/patriciogonzalezvivo/lygia_of_examples) 
* [2D/3D examples for Unity3D (HLSL)](https://github.com/patriciogonzalezvivo/lygia_unity_examples)
* [2D examples for Touch Designer (GLSL)](https://derivative.ca/community-post/asset/lygia-touchdesginer/66804) (dynamic resolver) by [Leith Ben Abdessalem](https://leithba.com)
* [2D examples for Touch Designer (GLSL)](https://github.com/vectorsize/lygia-td) (static resolver) by [Victor Saz](https://github.com/vectorsize)
* [2D examples on Observable Notebook (GLSL)](https://observablehq.com/@radames/hello-lygia-shader-library) by [Radames Ajna](https://twitter.com/radamar)
* [Figma's noise&texture plugin](https://www.figma.com/community/plugin/1138854718618193875) by [Rogie King](https://twitter.com/rogie). You will need to go to the "Custom" tab on the plugin to edit shaders and load LYGIA modules  
* [3D example on Irmf](https://github.com/irmf/irmf-examples/tree/master/examples/028-lygia) by [Glenn Lewis](https://github.com/gmlewis)
* [2D/3D examples on GlslViewer (GLSL)](https://github.com/patriciogonzalezvivo/lygia_examples)
* [2D examples on Ossia by Jean-Michaël Celerier](https://github.com/ossia/score-examples)

For more information, guidance or feedback about using LYGIA, join [#Lygia channel on shader.zone discord](https://shader.zone/).


### How is it organized?

The functions are divided in different categories:

* [`math/`](https://lygia.xyz/math): general math functions and constants like `PI`, `SqrtLength()`, etc. 
* [`space/`](https://lygia.xyz/space): general spatial operations like `scale()`, `rotate()`, etc. 
* [`color/`](https://lygia.xyz/color): general color operations like `luma()`, `saturation()`, blend modes, palettes, color space conversion and tonemaps.
* [`animation/`](https://lygia.xyz/animation): animation operations, like easing
* [`generative/`](https://lygia.xyz/generative): generative functions like `random()`, `noise()`, etc. 
* [`sdf/`](https://lygia.xyz/sdf): signed distance field functions.
* [`draw/`](https://lygia.xyz/draw): drawing functions like `digits()`, `stroke()`, `fill`, etc/.
* [`sample/`](https://lygia.xyz/sample): sample operations
* [`filter/`](https://lygia.xyz/filter): typical filter operations like different kind of blurs, mean and median filters.
* [`distort/`](https://lygia.xyz/distort): distort sampling operations
* [`simulate/`](https://lygia.xyz/simulate): simulate sampling operations
* [`lighting/`](https://lygia.xyz/lighting): different lighting models and functions for foward/deferred/raymarching rendering
* [`geometry/`](https://lygia.xyz/geometry): operation related to geometries. Like intersections and AABB accelerating structures.
* [`morphological/`](https://lygia.xyz/morphological): morphological filters like: dilation, erosion, alpha and poisson fill.

### Flexible how?

There are some functions whose behaviour can be changed using the `#defines` keyword before including it. For example, [gaussian blurs](filter/gaussianBlur.glsl) are usually are done in two passes. By default, these are performed on their 1D version, but in the case you are interested on using a 2D kernel, all in the same pass, you will need to add the `GAUSSIANBLUR_2D` keyword this way:

```glsl

    #define GAUSSIANBLUR_2D
    #include "filter/gaussianBlur.glsl"

    void main(void) {
        ...
        
        vec2 pixel = 1./u_resolution;
        color = gaussianBlur(u_tex0, uv, pixel, 9);
        ...
    }

```


## Design Principles

1. It relies on `#include "path/to/file.*lsl"` which is defined by Khronos GLSL standard and requires a tipical C-like pre-compiler MACRO which is easy to implement with just basic string operations to resolve dependencies. 

Here you can find some implementations on different languages:

  - C#:
  
    . [GLSLIncludes](https://github.com/seb776/GLSLIncludes) a small utility to add the include feature to glsl by [z0rg](https://github.com/seb776)

  - C++:

    . [VERA's routines](https://github.com/patriciogonzalezvivo/vera/blob/main/src/ops/fs.cpp#L110-L171) for resolving GLSL dependencies.

  - Python:

    . [Small and simple routing to resolve includes](https://gist.github.com/patriciogonzalezvivo/9a50569c2ef9b08058706443a39d838e)

  - JavaScript: 
  
    . [vanilla JS (online resolver)](https://lygia.xyz/resolve.js) This small file brings `resolveLygia()` which takes a `string` or `string[]` and parse it solving all the `#include` dependencies into a single `string` you can load on your shaders
  
    . [npm module (online resolver)](https://www.npmjs.com/package/resolve-lygia) by Eduardo Fossas; [vite glsl plugin (local bundle)](https://github.com/UstymUkhman/vite-plugin-glsl) by Ustym Ukhman. Imports `.glsl` local dependencies, or load inline shaders through vite
  
    . [esbuild glsl plugin (local bundle)](https://github.com/ricardomatias/esbuild-plugin-glsl-include) by Ricardo Matias. Import local `.glsl` dependencies through esbuild

    . [webpack glsl plugin (local bundle)](https://github.com/grieve/webpack-glsl-loader) by Ryan Grieve that import local `.glsl` dependencies through webpack.
        
* It's **very granular**. One function per file. The file and the function share the same name, namely: `myFunc.glsl` contains `myFunct()`. There are some files that just include a collection of files inside a folder with the same name. For example:

```
    color/blend.glsl
    // which includes
    color/blend/*.glsl

```

* It's **multi language**. Right now most of is GLSL (`*.glsl`) and HLSL (`*.hlsl`), but we are slowly extending to WGSL (`*.wgsl`), CUDA (`*.cuh`) and Metal (`*.msl`).

```
    math/mirror.glsl
    math/mirror.hlsl
    math/mirror.wgsl
    math/mirror.msl
    math/mirror.cuh
```

* **Self documented**. Each file contains a structured comment (in YAML) at the top of the file. This one contains the name of the original author, description, use and `#define` options

```glsl

    /*
    original_author: <FULL NAME>
    description: [DESCRIPTION + URL]
    use: <vec2> myFunc(<vec2> st, <float> x [, <float> y])
    options:
        - MYFUNC_TYPE
        - MYFUNC_SAMPLER_FNC()
    */

```

* Prevents **name collisions** by using the following pattern where `FNC_` is followed with the function name:

```glsl

    #ifndef FNC_MYFUNC
    #define FNC_MYFUNC

    float myFunc(float in) {
        return in;
    }

    #endif

```

* **Templating capabilities through `#defines`**. Probably the most frequent use is templating the sampling function for reusability. The `#define` options start with the name of the function, in this example `MYFUNC_`. They are added as `options:` in the header.
 
```glsl

    #ifndef MYFUNC_TYPE
    #define MYFUNC_TYPE vec4
    #endif

    #ifndef MYFUNC_SAMPLER_FNC
    #define MYFUNC_SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
    #endif

    #ifndef FNC_MYFUNC
    #define FNC_MYFUNC
    MYFUNC_TYPE myFunc(sampler2D tex, vec2 st) {
        return MYFUNC_SAMPLER_FNC(tex, st);
    }
    #endif

```

* **Function Overloading**. Arguments are arranged in such a way that optional elements are at the back. When possible sort them according their memory size (except textures that reamin at the top). Ex.: `sampler2D, mat4, mat3, mat2, vec4, vec3, vec2, float, ivec4, ivec3, ivec2, int, bool`

```glsl

    /*
    ...
    use: myFunc(<vec2> st, <vec2|float> x[, <float> y])
    */

    #ifndef FNC_MYFUNC
    #define FNC_MYFUNC
    vec2 myFunc(vec2 st, vec2 x) {
        return st * x;
    }

    vec2 myFunc(vec2 st, float x) {
        return st * x;
    }

    vec2 myFunc(vec2 st, float x, float y) {
        return st * vec2(x, y);
    }
    #endif

```

## Contributions

LYGIA have a long way to go. Your support will be appreciated and rewarded (all contributors are automatically added to the [commercial license](https://lygia.xyz/license) ). This support can take multiple forms:

* fixing bugs!
* expanding the crosscompatibility between languages GLSL/HLSL/MSL/WGSL/CUDA
* contributing new functions
* adding new examples and integrations for new enviroments like: [GoDot](https://godotengine.org/), [ISF](https://isf.video/), [MaxMSP](https://cycling74.com/products/max), etc.
* through [sponsorships](https://github.com/sponsors/patriciogonzalezvivo)


## License 

LYGIA is dual-licensed under [the Prosperity License](https://prosperitylicense.com/versions/3.0.0) and the [Patron License](https://lygia.xyz/license) for [sponsors](https://github.com/sponsors/patriciogonzalezvivo) and [contributors](https://github.com/patriciogonzalezvivo/lygia/graphs/contributors).

[Sponsors](https://github.com/sponsors/patriciogonzalezvivo) and [contributors](https://github.com/patriciogonzalezvivo/lygia/graphs/contributors) are automatically added to the [Patron License](https://lygia.xyz/license) and they can ignore the any non-commercial rule of [the Prosperity Licensed](https://prosperitylicense.com/versions/3.0.0) software (please take a look to the exception).

It's also possible to get a permanent comercial license hook to a single and specific version of LYGIA.

**Exceptions**:

* `color/mixBox.glsl` and `color/mixBox.hlsl` it's copyrighted by Secret Weapons with their own non-commercial license. This functions also require a LUT texture wich is provided for research and evaluation purposes, if you wish to obtain it together with a commercial license, please contact them at mixbox@scrtwpns.com 

## Credits

Created and mantain by [Patricio Gonzalez Vivo](https://patriciogonzalezvivo.com/)( <a rel="me" href="https://merveilles.town/@patricio">Mastodon</a> | [Twitter](https://twitter.com/patriciogv) | [Instagram](https://www.instagram.com/patriciogonzalezvivo/) | [GitHub](https://github.com/sponsors/patriciogonzalezvivo) ) and every direct or indirect [contributors](https://github.com/patriciogonzalezvivo/lygia/graphs/contributors) to the GitHub. This library has been built over years, and in many cases on top of the work of brillant and generous people like: [Inigo Quiles](https://www.iquilezles.org/), [Morgan McGuire](https://casual-effects.com/), [Alan Wolfe](https://blog.demofox.org/), [Hugh Kennedy](https://github.com/hughsk), [Matt DesLauriers](https://www.mattdesl.com/) and many others.


## Get the latest news and releases

Sign up for the news letter bellow, joing [the LYGIA's channel on Discord](https://shader.zone) or follow the [Github repository](https://github.com/patriciogonzalezvivo/lygia)
