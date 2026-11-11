// Run once in each script's own context: locks the context down, then
// returns the function that builds a script's runner from the host's
// callback, the script's function and its fields.
(function () {
  "use strict";
  // Memory outside the heap V8 limits, and objects that tell a script about
  // the collector: gone before any script runs. The host makes its own typed
  // arrays through V8's API.
  for (const name of [
    "ArrayBuffer",
    "SharedArrayBuffer",
    "DataView",
    "Atomics",
    "Int8Array",
    "Uint8Array",
    "Uint8ClampedArray",
    "Int16Array",
    "Uint16Array",
    "Int32Array",
    "Uint32Array",
    "Float16Array",
    "Float32Array",
    "Float64Array",
    "BigInt64Array",
    "BigUint64Array",
    "Intl",
    "WeakRef",
    "FinalizationRegistry",
    "WebAssembly",
  ]) {
    delete globalThis[name];
  }
  // The last match, which RegExp's legacy accessors read and set across
  // calls.
  for (const name of [
    "input",
    "$_",
    "lastMatch",
    "$&",
    "lastParen",
    "$+",
    "leftContext",
    "$`",
    "rightContext",
    "$'",
    "$1",
    "$2",
    "$3",
    "$4",
    "$5",
    "$6",
    "$7",
    "$8",
    "$9",
  ]) {
    delete RegExp[name];
  }

  // Every object a script can reach freezes: what it holds, its accessors and
  // its prototype, transitively, so no script changes what a later document
  // or request sees.
  const freeze = Object.freeze;
  const ownKeys = Reflect.ownKeys;
  const describe = Object.getOwnPropertyDescriptor;
  const protoOf = Object.getPrototypeOf;
  function harden(root) {
    const seen = new Set();
    const todo = [root];
    while (todo.length > 0) {
      const o = todo.pop();
      if (o === null || (typeof o !== "object" && typeof o !== "function") || seen.has(o)) {
        continue;
      }
      seen.add(o);
      freeze(o);
      for (const k of ownKeys(o)) {
        const d = describe(o, k);
        if (d === undefined) continue;
        if ("value" in d) todo.push(d.value);
        if (d.get !== undefined) todo.push(d.get);
        if (d.set !== undefined) todo.push(d.set);
      }
      todo.push(protoOf(o));
    }
    return root;
  }

  // The intrinsics no global names: the prototypes of functions, iterators
  // and generators a script can make.
  const intrinsics = [];
  const reach = (make) => {
    try {
      intrinsics.push(make());
    } catch (e) {
      // An engine without it.
    }
  };
  reach(() => function* () {});
  reach(() => async function () {});
  reach(() => async function* () {});
  reach(() => (async function* () {})());
  reach(() => (function* () {})());
  reach(() => [][Symbol.iterator]());
  reach(() => new Map()[Symbol.iterator]());
  reach(() => new Set()[Symbol.iterator]());
  reach(() => ""[Symbol.iterator]());
  reach(() => /a/[Symbol.matchAll]("a"));
  reach(() => [].values().map((x) => x));
  reach(() => Iterator.from({ next() { return { done: true }; } }));
  // An error's own stack accessor.
  reach(() => new Error());
  reach(() => {
    try {
      null.x;
    } catch (e) {
      return e;
    }
  });
  for (const o of intrinsics) harden(o);
  harden(globalThis);

  const NO_VALUE =
    "A document doesn't have a value for a field! Use doc[<field>].size()==0 to check if a document is missing a field!";

  // need(f): stops the call for the host to load field f, after which the
  // document runs again. cols[f]: field f's column for the call's documents,
  // as [values, starts, terms] (document k's are values[starts[k] ..
  // starts[k + 1]], a keyword's terms beside its ordinals), or undefined
  // until loaded. kinds: 0 a number, 1 a boolean, 2 a keyword, 3 a mapped
  // field no script can read. state[0]: the document running.
  return function (need, user, names, kinds, cols, state) {
    const n = names.length;
    let at = 0;
    // Whether the document running asked for a field: its result is
    // dropped, whatever the script made of the error.
    let faulted = false;

    function column(f) {
      const c = cols[f];
      if (c === undefined) {
        faulted = true;
        need(f);
        // The host stops the call before this returns to the script.
        throw new Error("the field is not loaded");
      }
      return c;
    }

    function count(f) {
      const s = column(f)[1];
      return s[at + 1] - s[at];
    }

    function item(f, k) {
      const c = column(f);
      const s = c[1];
      const from = s[at];
      const len = s[at + 1] - from;
      if (len === 0) throw new Error(NO_VALUE);
      if (!(k >= 0 && k < len)) {
        throw new RangeError(`Index ${k} out of bounds for length ${len}`);
      }
      switch (kinds[f]) {
        case 1:
          return c[0][from + k] !== 0;
        case 2:
          return c[2][from + k];
        default:
          return c[0][from + k];
      }
    }

    // Only the runner makes a field.
    const MAKER = {};

    // doc['f']: its values in the document running.
    class Field {
      #f;
      constructor(maker, f) {
        if (maker !== MAKER) throw new TypeError("Illegal constructor");
        this.#f = f;
        freeze(this);
      }
      get value() {
        return item(this.#f, 0);
      }
      get(k) {
        return item(this.#f, k);
      }
      size() {
        return count(this.#f);
      }
      get length() {
        return count(this.#f);
      }
      get empty() {
        return count(this.#f) === 0;
      }
      isEmpty() {
        return count(this.#f) === 0;
      }
      get values() {
        const c = count(this.#f);
        const out = [];
        for (let k = 0; k < c; k++) out.push(item(this.#f, k));
        return out;
      }
      [Symbol.iterator]() {
        return this.values[Symbol.iterator]();
      }
    }

    // doc['f'] for a field without doc values a script can read.
    function unreadable(name) {
      const fail = () => {
        throw new Error(`doc[${name}]: the field has no doc values a js script can read`);
      };
      return {
        get value() {
          return fail();
        },
        get: fail,
        size: fail,
        get length() {
          return fail();
        },
        get empty() {
          return fail();
        },
        isEmpty: fail,
        get values() {
          return fail();
        },
      };
    }

    const base = {
      containsKey(name) {
        return Object.prototype.hasOwnProperty.call(doc, name);
      },
    };
    // Any other name: Painless's error for a field the mapping lacks.
    const missing = new Proxy(base, {
      get(target, key, receiver) {
        if (typeof key === "symbol" || key in target || key === "toJSON" || key === "then") {
          return Reflect.get(target, key, receiver);
        }
        throw new Error(`No field found for [${key}] in mapping`);
      },
    });
    const doc = Object.create(missing);
    for (let f = 0; f < n; f++) {
      Object.defineProperty(doc, names[f], {
        value: kinds[f] === 3 ? unreadable(names[f]) : new Field(MAKER, f),
        enumerable: true,
      });
    }
    harden(base);
    harden(Field);
    harden(doc);
    freeze(missing);

    function what(r) {
      return r === null ? "null" : Array.isArray(r) ? "an array" : typeof r;
    }

    return freeze({
      // Documents from .. count of the call into out, as numbers (mode 0) or
      // booleans as 1 and 0 (mode 1).
      batch(from, count, out, scores, params, mode) {
        faulted = false;
        for (at = from; at < count; at++) {
          state[0] = at;
          const r = user(doc, params, scores === null ? undefined : scores[at]);
          if (faulted) return;
          if (mode === 1) {
            if (typeof r !== "boolean") {
              throw new TypeError(`a filter script must return a boolean, not ${what(r)}`);
            }
            out[at] = r ? 1 : 0;
          } else {
            if (typeof r !== "number") {
              throw new TypeError(`a score script must return a number, not ${what(r)}`);
            }
            out[at] = r;
          }
        }
      },
      // The call's one document's value, as the script returns it.
      one(score, params) {
        faulted = false;
        at = 0;
        state[0] = 0;
        const r = user(doc, params, score);
        return faulted ? undefined : r;
      },
      // A request's params, as scripts read them.
      params(p) {
        return harden(p);
      },
    });
  };
})();
