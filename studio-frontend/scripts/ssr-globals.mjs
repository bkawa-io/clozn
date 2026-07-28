/**
 * Minimal browser globals so the Studio's module graph can be imported and rendered under Node.
 *
 * Imported for its side effects BEFORE anything that touches these -- ES module imports execute in
 * source order, which is what guarantees the stubs exist by the time App.tsx's module body runs.
 *
 * Deliberately minimal and deliberately NOT jsdom: the goal is to prove the component tree renders,
 * not to simulate a browser. Anything a panel calls that is missing here should surface as a loud
 * failure in the smoke render, because it means that panel is doing browser work at module scope or
 * during first render -- both of which are worth knowing about.
 *
 * `renderToString` does not run effects, so `useEffect` bodies (data loading, hash listeners,
 * `useTopbar` publication, the theme write to localStorage) never execute here.
 */
const store = new Map();

globalThis.localStorage = {
  getItem: (key) => (store.has(key) ? store.get(key) : null),
  setItem: (key, value) => void store.set(key, String(value)),
  removeItem: (key) => void store.delete(key),
  clear: () => store.clear(),
};

globalThis.location = { hash: "", href: "http://127.0.0.1/", pathname: "/", search: "" };

globalThis.history = {
  replaceState: () => {},
  pushState: () => {},
};

globalThis.document = {
  documentElement: { dataset: {} },
};

globalThis.window = globalThis;
globalThis.addEventListener = () => {};
globalThis.removeEventListener = () => {};

// TraceScope reads a media query during its first render (features/observatory/TraceScope.tsx:270).
// It guards with `typeof window !== "undefined"`, which passes here because `window` is stubbed above,
// so matchMedia has to exist too. Reports "not narrow" -- the desktop layout, which is the one worth
// smoke-rendering.
globalThis.matchMedia = () => ({
  matches: false,
  media: "",
  addEventListener: () => {},
  removeEventListener: () => {},
  addListener: () => {},
  removeListener: () => {},
  onchange: null,
  dispatchEvent: () => false,
});
