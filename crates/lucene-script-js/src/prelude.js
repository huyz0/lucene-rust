// Run once in each script's own context: locks the context's built-ins, then
// returns the function that builds a script's runner from the host's
// callbacks, the script's function and its fields.
(function () {
  "use strict";
  // No script changes what a later document or request sees: every built-in
  // and its prototype is frozen, and so is the global object.
  for (const name of Object.getOwnPropertyNames(globalThis)) {
    const v = globalThis[name];
    if (v !== null && (typeof v === "object" || typeof v === "function")) {
      Object.freeze(v);
      if (typeof v === "function" && v.prototype) Object.freeze(v.prototype);
    }
  }
  for (const p of [
    Object.getPrototypeOf(function* () {}),
    Object.getPrototypeOf(async function () {}),
    Object.getPrototypeOf(Object.getPrototypeOf([][Symbol.iterator]())),
    Object.getPrototypeOf(new Map()[Symbol.iterator]()),
    Object.getPrototypeOf(new Set()[Symbol.iterator]()),
    Object.getPrototypeOf(""[Symbol.iterator]()),
    Object.getPrototypeOf([][Symbol.iterator]()),
    Object.getPrototypeOf(Int8Array.prototype),
    Object.getPrototypeOf(Int8Array),
  ]) {
    Object.freeze(p);
  }
  Object.freeze(globalThis);

  const NO_VALUE =
    "A document doesn't have a value for a field! Use doc[<field>].size()==0 to check if a document is missing a field!";
  const MAX_TERMS = 4096;

  // load(f): field f's values for the call's documents, as [values, starts]
  // (document k's are values[starts[k] .. starts[k + 1]]). term(f, ord): a
  // keyword's term. kinds: 0 a number, 1 a boolean, 2 a keyword (its
  // ordinals), 3 a mapped field no script can read. state[0]: the document
  // running, for an error.
  return function (load, term, user, names, kinds, state) {
    const n = names.length;
    const vals = new Array(n).fill(null);
    const starts = new Array(n).fill(null);
    const loaded = new Float64Array(n).fill(-1);
    const terms = [];
    for (let f = 0; f < n; f++) terms.push(new Map());
    let epoch = 0;
    let at = 0;
    let segment = -1;

    function column(f) {
      if (loaded[f] !== epoch) {
        const c = load(f);
        vals[f] = c[0];
        starts[f] = c[1];
        loaded[f] = epoch;
      }
    }

    function count(f) {
      column(f);
      const s = starts[f];
      return s[at + 1] - s[at];
    }

    function item(f, k) {
      column(f);
      const s = starts[f];
      const from = s[at];
      const len = s[at + 1] - from;
      if (len === 0) throw new Error(NO_VALUE);
      if (!(k >= 0 && k < len)) {
        throw new RangeError(`Index ${k} out of bounds for length ${len}`);
      }
      const v = vals[f][from + k];
      switch (kinds[f]) {
        case 1:
          return v !== 0;
        case 2: {
          const m = terms[f];
          let t = m.get(v);
          if (t === undefined) {
            t = term(f, v);
            if (m.size >= MAX_TERMS) m.clear();
            m.set(v, t);
          }
          return t;
        }
        default:
          return v;
      }
    }

    // doc['f']: its values in the document running.
    class Field {
      constructor(f) {
        this.f = f;
        Object.freeze(this);
      }
      get value() {
        return item(this.f, 0);
      }
      get(k) {
        return item(this.f, k);
      }
      size() {
        return count(this.f);
      }
      get length() {
        return count(this.f);
      }
      get empty() {
        return count(this.f) === 0;
      }
      isEmpty() {
        return count(this.f) === 0;
      }
      get values() {
        const c = count(this.f);
        const out = [];
        for (let k = 0; k < c; k++) out.push(item(this.f, k));
        return out;
      }
      [Symbol.iterator]() {
        return this.values[Symbol.iterator]();
      }
    }
    Object.freeze(Field.prototype);
    Object.freeze(Field);

    // doc['f'] for a field without doc values a script can read.
    function unreadable(name) {
      const fail = () => {
        throw new Error(`doc[${name}]: the field has no doc values a js script can read`);
      };
      return Object.freeze({
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
      });
    }

    const base = Object.freeze({
      containsKey(name) {
        return Object.prototype.hasOwnProperty.call(doc, name);
      },
    });
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
        value: kinds[f] === 3 ? unreadable(names[f]) : new Field(f),
        enumerable: true,
      });
    }
    Object.freeze(doc);

    function what(r) {
      return r === null ? "null" : Array.isArray(r) ? "an array" : typeof r;
    }

    function enter(seg) {
      epoch++;
      if (seg !== segment) {
        segment = seg;
        for (const m of terms) m.clear();
      }
    }

    return Object.freeze({
      // Documents 0 .. count of the call into out, as numbers (mode 0) or
      // booleans as 1 and 0 (mode 1).
      batch(count, out, scores, params, mode, seg) {
        enter(seg);
        for (at = 0; at < count; at++) {
          state[0] = at;
          const r = user(doc, params, scores === null ? undefined : scores[at]);
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
      one(score, params, seg) {
        enter(seg);
        at = 0;
        state[0] = 0;
        return user(doc, params, score);
      },
    });
  };
})();
