const path = require("node:path");

function readPackage(pkg) {
  if (process.env.LOCAL_DEPS) {
    const toolsPkgs = path.resolve(__dirname, "../wesl-js/tools/packages");
    pkg.dependencies = {
      ...pkg.dependencies,
      "wesl-link": `link:${toolsPkgs}/wesl-link`,
      "wesl-packager": `link:${toolsPkgs}/wesl-packager`,
      "wgsl-test": `link:${toolsPkgs}/wgsl-test`,
    };
  }
  return pkg;
}

module.exports = { hooks: { readPackage } };
