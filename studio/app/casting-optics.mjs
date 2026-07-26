/* Clozn Studio — luminous optical scaffold for THE CASTING.
   Decorative only: this overlay does not encode model measurements.
   All measured content remains in casting.mjs beneath it.

   Usage:
     const optics = mountCastingOptics(container);
     ...
     optics.setActivity(0.7);
     optics.destroy();
*/

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function cssColor(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function rgba(hex, alpha) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return `rgba(180,180,240,${alpha})`;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

export function mountCastingOptics(container, opts = {}) {
  if (!container) return { setActivity() {}, destroy() {} };

  const root = document.createElement("div");
  root.className = "casting-optics";
  root.setAttribute("aria-hidden", "true");
  root.innerHTML = `
    <canvas class="casting-optics-canvas"></canvas>
    <div class="casting-optics-register">
      <span>LIVE CAST</span>
      <i></i>
      <span class="casting-optics-mode">OPTICAL FIELD</span>
    </div>
    <div class="casting-optics-depth">
      <span>INPUT</span><span>SHALLOW</span><span>MIDDLE</span><span>DEEP</span><span>OUTPUT</span>
    </div>
    <div class="casting-optics-corners">
      <b></b><b></b><b></b><b></b>
    </div>
  `;
  container.appendChild(root);

  const canvas = root.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let activity = clamp(Number(opts.activity ?? 0.35), 0, 1);
  let targetX = 0, targetY = 0, pointerX = 0, pointerY = 0;
  let w = 1, h = 1, dpr = 1, raf = 0, dead = false;
  let palette = {};

  function readPalette() {
    palette = {
      peri: cssColor("--iri-peri", "#96AFF2"),
      mint: cssColor("--iri-mint", "#6ECDBE"),
      pink: cssColor("--iri-pink", "#E896D6"),
      lilac: cssColor("--iri-lilac", "#BAB2E2"),
      gold: cssColor("--iri-gold", "#F0D096"),
      ink: cssColor("--ink", "#E8E9F8"),
      line: cssColor("--line-strong", "#3B4070"),
    };
  }

  function resize() {
    const r = container.getBoundingClientRect();
    w = Math.max(1, Math.round(r.width));
    h = Math.max(1, Math.round(r.height));
    dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function pathPlane(cx, cy, pw, ph, skew, depth, color, time) {
    const pulse = reduce ? 0 : Math.sin(time * 0.0005 + depth * 1.4) * 0.5 + 0.5;
    const ox = pointerX * (5 + depth * 2);
    const oy = pointerY * (3 + depth);
    const x = cx + ox;
    const y = cy + oy;
    ctx.save();
    ctx.translate(x, y);
    ctx.transform(1, 0, skew, 1, 0, 0);

    const grad = ctx.createLinearGradient(-pw / 2, -ph / 2, pw / 2, ph / 2);
    grad.addColorStop(0, rgba(color, 0.015));
    grad.addColorStop(0.48, rgba(color, 0.035 + pulse * 0.018));
    grad.addColorStop(1, rgba(color, 0.008));
    ctx.fillStyle = grad;
    ctx.strokeStyle = rgba(color, 0.12 + activity * 0.08);
    ctx.lineWidth = 1;
    ctx.shadowColor = rgba(color, 0.18 + activity * 0.16);
    ctx.shadowBlur = 8 + activity * 8;
    ctx.beginPath();
    ctx.roundRect(-pw / 2, -ph / 2, pw, ph, 8);
    ctx.fill();
    ctx.stroke();

    // Decorative matrix: deliberately abstract, never a measurement.
    ctx.shadowBlur = 0;
    const cols = 7, rows = 13;
    for (let yy = 0; yy < rows; yy++) {
      for (let xx = 0; xx < cols; xx++) {
        const seed = Math.sin((xx + 1) * 12.31 + (yy + 1) * 7.17 + depth * 3.1);
        if (seed < -0.26) continue;
        const a = 0.025 + (seed + 1) * 0.025;
        ctx.fillStyle = rgba(color, a);
        const px = -pw * 0.29 + xx * (pw * 0.58 / (cols - 1));
        const py = -ph * 0.32 + yy * (ph * 0.64 / (rows - 1));
        ctx.fillRect(px, py, 1.4 + depth * 0.15, 1.4 + depth * 0.15);
      }
    }
    ctx.restore();
  }

  function filament(x1, y1, x2, y2, bend, color, alpha, time, index) {
    const pulse = reduce ? 0.55 : 0.42 + 0.18 * Math.sin(time * 0.0012 + index * 0.7);
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.bezierCurveTo(mx - bend, my - 16, mx + bend, my + 16, x2, y2);
    ctx.strokeStyle = rgba(color, alpha * pulse);
    ctx.lineWidth = 0.7 + activity * 0.65;
    ctx.shadowColor = rgba(color, 0.26 + activity * 0.18);
    ctx.shadowBlur = 4 + activity * 7;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  function draw(time = 0) {
    if (dead) return;
    pointerX += (targetX - pointerX) * 0.055;
    pointerY += (targetY - pointerY) * 0.055;
    ctx.clearRect(0, 0, w, h);

    const cx = w * 0.5;
    const cy = h * 0.50;
    const count = w < 760 ? 5 : 6;
    const span = Math.min(w * 0.78, 930);
    const step = span / (count - 1);
    const start = cx - span / 2;
    const planeW = clamp(step * 0.62, 58, 118);
    const planeH = clamp(h * 0.59, 250, 510);
    const colors = [palette.peri, palette.mint, palette.pink, palette.lilac, palette.mint, palette.peri];

    ctx.globalCompositeOperation = "lighter";

    for (let i = 0; i < count; i++) {
      const depth = i / Math.max(1, count - 1);
      const x = start + step * i;
      const y = cy + Math.sin(depth * Math.PI * 2) * 7;
      pathPlane(x, y, planeW, planeH, -0.10 + depth * 0.20, depth, colors[i], time);
    }

    // Energy crossings between planes. They frame, rather than replace, the real cast.
    for (let i = 0; i < count - 1; i++) {
      const x1 = start + step * i + planeW * 0.34;
      const x2 = start + step * (i + 1) - planeW * 0.34;
      for (let k = 0; k < 11; k++) {
        const u = k / 10;
        const y1 = cy + (u - 0.5) * planeH * 0.42;
        const y2 = cy + (0.5 - u) * planeH * 0.32;
        const c = i % 2 ? palette.pink : palette.peri;
        filament(x1, y1, x2, y2, 18 + Math.sin(k * 1.3) * 14, c, 0.15 + activity * 0.12, time, i * 20 + k);
      }
    }

    // Central optical axis.
    const axis = ctx.createLinearGradient(0, h * 0.15, 0, h * 0.84);
    axis.addColorStop(0, rgba(palette.mint, 0));
    axis.addColorStop(0.46, rgba(palette.mint, 0.10 + activity * 0.10));
    axis.addColorStop(0.54, rgba(palette.pink, 0.13 + activity * 0.11));
    axis.addColorStop(1, rgba(palette.pink, 0));
    ctx.fillStyle = axis;
    ctx.shadowColor = rgba(palette.lilac, 0.22 + activity * 0.18);
    ctx.shadowBlur = 14 + activity * 20;
    ctx.fillRect(cx - 0.5, h * 0.15, 1, h * 0.69);
    ctx.shadowBlur = 0;

    ctx.globalCompositeOperation = "source-over";
    if (!reduce) raf = requestAnimationFrame(draw);
  }

  function onPointer(e) {
    const r = container.getBoundingClientRect();
    targetX = clamp((e.clientX - r.left) / Math.max(1, r.width) - 0.5, -0.5, 0.5);
    targetY = clamp((e.clientY - r.top) / Math.max(1, r.height) - 0.5, -0.5, 0.5);
  }
  function leave() { targetX = 0; targetY = 0; }

  const resizeObserver = new ResizeObserver(() => { resize(); if (reduce) draw(0); });
  resizeObserver.observe(container);
  container.addEventListener("pointermove", onPointer, { passive: true });
  container.addEventListener("pointerleave", leave);

  const mutationObserver = new MutationObserver(() => { readPalette(); if (reduce) draw(0); });
  mutationObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  readPalette();
  resize();
  if (reduce) draw(0);
  else raf = requestAnimationFrame(draw);

  return {
    setActivity(value) {
      activity = clamp(Number(value) || 0, 0, 1);
      root.style.setProperty("--casting-activity", activity);
      if (reduce) draw(0);
    },
    destroy() {
      dead = true;
      cancelAnimationFrame(raf);
      resizeObserver.disconnect();
      mutationObserver.disconnect();
      container.removeEventListener("pointermove", onPointer);
      container.removeEventListener("pointerleave", leave);
      root.remove();
    },
  };
}
