import path from 'path'
const __dirname = path.resolve()

const config = {
  mode: 'production',
  entry: {
    main: './index.js',
    animation: './animation',
    color: './color',
    distort: './distort',
    draw: './draw',
    filter: './filter',
    generative: './generative',
    geometry: './geometry',
    lighting: './lighting',
    math: './math',
    morphological: './morphological',
    sample: './sample',
    sdf: './sdf',
    simulate: './simulate',
    space: './space',
  },
  module: {
    rules: [
      {
        test: /\.glsl$/,
        use: {
          loader: 'webpack-glsl-loader',
        },
      },
    ],
  },
}
const targetModule = {
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'module/lygia.[name].js',
    libraryTarget: 'module',
  },
  experiments: {
    outputModule: true,
  },
}
const targetUmd = {
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'umd/lygia.[name].js',
    library: {
      root: 'Lygia',
      amd: 'lygia',
      commonjs: 'lygia',
    },
    libraryTarget: 'umd',
  },
}
export default [
  Object.assign({}, config, targetUmd),
  Object.assign({}, config, targetModule),
]