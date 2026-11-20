// Preloaded into every test process by run-tests.mjs. A test whose runtime
// launch throws `VrUnavailableError` (a `--vr` launch on the flatscreen-only
// branch) is reported as skipped instead of failed.
import { createRequire, syncBuiltinESMExports } from "node:module";

const nodeTest = createRequire(import.meta.url)("node:test");

function skipVr(fn) {
  // Leave callback-style tests (t, done) alone.
  if (typeof fn !== "function" || fn.length >= 2) return fn;
  return async function (t) {
    try {
      return await fn.call(this, t);
    } catch (error) {
      if (error?.name !== "VrUnavailableError") throw error;
      t.skip(error.message);
    }
  };
}

function wrap(original) {
  const wrapped = function (...args) {
    const at = args.findLastIndex((arg) => typeof arg === "function");
    if (at >= 0) args[at] = skipVr(args[at]);
    return original.apply(this, args);
  };
  for (const key of ["skip", "todo", "only"]) {
    if (original[key]) wrapped[key] = wrap(original[key]);
  }
  return wrapped;
}

nodeTest.test = wrap(nodeTest.test);
nodeTest.it = wrap(nodeTest.it);
syncBuiltinESMExports();
