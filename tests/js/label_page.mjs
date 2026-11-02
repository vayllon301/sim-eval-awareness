// Drives the labelling page (src/simtest/judge/label_page.html) in jsdom against a running
// label server, and prints what happened as JSON for tests/test_judge_labels.py.
//
//     node tests/js/label_page.mjs <base url> <scenario>
//
// The page's fetch goes to the server through a wrapper that logs every request and can
// hold one back until the scenario releases it. Prints no transcript text.
import { JSDOM } from "jsdom";

const [base, scenario] = process.argv.slice(2);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function until(cond, what, ms = 5000) {
  const end = Date.now() + ms;
  while (!cond()) {
    if (Date.now() > end) throw new Error(`timed out waiting for ${what}`);
    await sleep(5);
  }
}

const requests = [];       // {method, path, label_id, fields, keepalive}
let hold = null;           // {method, path, release, seen}

function holdNext(method, path) {
  let release;
  const gate = new Promise((r) => { release = r; });
  hold = { method, path, gate, release, seen: false };
  return hold;
}

async function pageFetch(url, opt = {}) {
  const u = new URL(url, base);
  const method = opt.method || "GET";
  const body = opt.body ? JSON.parse(opt.body) : {};
  requests.push({ method, path: u.pathname, label_id: body.label_id ?? null, fields: body.fields ?? null,
                  keepalive: !!opt.keepalive });
  if (hold && !hold.seen && hold.method === method && hold.path === u.pathname) {
    hold.seen = true;
    await hold.gate;
  }
  return fetch(u, opt);
}

const html = await (await fetch(new URL("/", base))).text();
const dom = new JSDOM(html, {
  url: new URL("/", base).href, runScripts: "dangerously", pretendToBeVisual: true,
  beforeParse(w) { w.fetch = pageFetch; w.scrollTo = () => {}; },
});
const w = dom.window, d = w.document, $ = (id) => d.getElementById(id);
const status = () => $("status").textContent;
const posts = () => requests.filter((r) => r.method === "POST");

// What a user can do: a disabled control takes no typing (click() on one is already a no-op).
function type(el, text) {
  if (el.disabled) return false;
  el.value = text;
  el.dispatchEvent(new w.Event("input", { bubbles: true }));
  return true;
}
function unload() {
  const ev = new w.Event("beforeunload", { cancelable: true });
  w.dispatchEvent(ev);
  return ev.defaultPrevented;
}

await until(() => $("lid").textContent !== "…" && status() !== "Loading…", "the first episode");
const out = { first: $("lid").textContent };

if (scenario === "edit-while-loading") {
  // Next is clicked; while the next episode loads, the labeller types a note on the current one.
  const h = holdNext("GET", "/api/episode/1");
  $("next").click();
  await until(() => h.seen, "the episode request");
  out.typed = type($("note"), "typed while the next episode loads");
  h.release();
  await until(() => $("lid").textContent !== out.first && status() !== "Loading…", "the next episode");
  await sleep(900);   // past the note's autosave delay
  out.second = $("lid").textContent;
  out.note_shown = $("note").value;
  out.enabled_after = [...d.querySelectorAll("#form input, #form select, #form textarea, #form button")]
    .every((c) => !c.disabled) && !$("prev").disabled && !$("jump").disabled;
} else if (scenario === "close-while-saving") {
  // An answer is saved; the tab is closed before the server acknowledges it.
  const h = holdNext("POST", "/api/label");
  d.querySelector("input[name=verdict][value=EVAL]").click();
  await until(() => h.seen, "the save request");
  out.prevented_while_saving = unload();
  h.release();
  await until(() => /^Saved /.test(status()), "the acknowledgement");
  out.prevented_after = unload();
} else if (scenario === "real-then-eval") {
  // REAL fills the unset onset with "none"; changing the verdict to EVAL must not keep it.
  d.querySelector("input[name=verdict][value=REAL]").click();
  out.onset_after_real = $("onset").value;
  d.querySelector("input[name=verdict][value=EVAL]").click();
  out.onset_after_eval = $("onset").value;
  await until(() => /^Saved /.test(status()) && posts().at(-1)?.fields?.verdict === "EVAL", "the save of EVAL");
} else if (scenario === "arrows-while-selecting") {
  $("gclose").click();   // the guide opens on a first visit and takes the keys while open
  const key = (k, shift) => d.body.dispatchEvent(new w.KeyboardEvent("keydown", { key: k, shiftKey: !!shift, bubbles: true }));
  const select = () => {
    const box = d.querySelector("[data-step] [data-part]");
    const r = d.createRange(); r.selectNodeContents(box);
    const s = w.getSelection(); s.removeAllRanges(); s.addRange(r);
  };
  select();
  key("ArrowRight", true);
  await sleep(300);
  out.after_shift_arrow = $("lid").textContent;
  key("ArrowRight");
  await sleep(300);
  out.after_arrow_with_selection = $("lid").textContent;
  w.getSelection().removeAllRanges();
  key("ArrowRight");
  await until(() => $("lid").textContent !== out.first && status() !== "Loading…", "the next episode");
  out.after_plain_arrow = $("lid").textContent;
} else {
  throw new Error(`unknown scenario ${scenario}`);
}

out.posts = posts();
w.close();
console.log(JSON.stringify(out));
