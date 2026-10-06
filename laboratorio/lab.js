// Utilidades compartidas del laboratorio: azar con semilla, lienzos nítidos, ejes, pestañas.
"use strict";

// Generador con semilla (mulberry32): mismos datos en cada computador.
function azar(semilla) {
  let a = semilla >>> 0;
  const u = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  u.normal = () => { let x = 0, y = 0; while (x === 0) x = u(); y = u(); return Math.sqrt(-2 * Math.log(x)) * Math.cos(2 * Math.PI * y); };
  u.entre = (a0, b0) => a0 + (b0 - a0) * u();
  return u;
}

// Lienzo con resolución de pantalla (nítido en celulares) y relación de aspecto fija.
function lienzo(id, aspecto = 0.68) {
  const c = document.getElementById(id), ctx = c.getContext("2d");
  const obj = { c, ctx, w: 0, h: 0, alDibujar: null };
  const ajustar = () => {
    const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = Math.round(w * aspecto);
    c.style.height = h + "px"; c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); obj.w = w; obj.h = h;
    if (obj.alDibujar) obj.alDibujar();
  };
  new ResizeObserver(ajustar).observe(c);
  obj.ajustar = ajustar;
  return obj;
}

// Escala lineal y ejes sencillos.
function escala(d0, d1, r0, r1) { const f = (v) => r0 + (v - d0) / (d1 - d0) * (r1 - r0); f.inv = (p) => d0 + (p - r0) / (r1 - r0) * (d1 - d0); return f; }
const COL = { texto: "#E8EDF6", suave: "#A9B5CC", tenue: "#6F7C96", rejilla: "rgba(255,255,255,.06)", eje: "rgba(255,255,255,.22)",
  m1: "#4C8DDB", m2: "#3BB27A", m3: "#F0953A", m4: "#9B6FD9", azul: "#4C8DDB", rojo: "#E0645A",
  grupos: ["#4C8DDB", "#F0953A", "#3BB27A", "#9B6FD9", "#E0645A", "#E8C547", "#4FC3C9", "#D96FB0"] };

function ejes(L, sx, sy, o = {}) {
  const { ctx } = L, m = L.margen;
  ctx.save(); ctx.font = "11px Inter, system-ui, sans-serif"; ctx.fillStyle = COL.tenue; ctx.strokeStyle = COL.rejilla; ctx.lineWidth = 1;
  (o.xt || []).forEach((v) => { const x = sx(v); ctx.beginPath(); ctx.moveTo(x, m.t); ctx.lineTo(x, L.h - m.b); ctx.stroke();
    ctx.textAlign = "center"; ctx.fillText(o.fx ? o.fx(v) : v, x, L.h - m.b + 15); });
  (o.yt || []).forEach((v) => { const y = sy(v); ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(L.w - m.r, y); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(o.fy ? o.fy(v) : v, m.l - 6, y + 4); });
  ctx.fillStyle = COL.suave; ctx.font = "12px Inter, system-ui, sans-serif";
  if (o.xl) { ctx.textAlign = "center"; ctx.fillText(o.xl, (m.l + L.w - m.r) / 2, L.h - 4); }
  if (o.yl) { ctx.save(); ctx.translate(12, (m.t + L.h - m.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText(o.yl, 0, 0); ctx.restore(); }
  ctx.restore();
}
function punto(ctx, x, y, r, relleno, borde) {
  ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); if (relleno) { ctx.fillStyle = relleno; ctx.fill(); }
  if (borde) { ctx.strokeStyle = borde; ctx.lineWidth = 1.5; ctx.stroke(); }
}
function rango(a, b, paso) { const r = []; for (let v = a; v <= b + 1e-9; v += paso) r.push(+v.toFixed(6)); return r; }
const fmt = (v, d = 2) => Number(v).toLocaleString("es-CO", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 0) => fmt(100 * v, d) + " %";

// Pestañas Mira / Juega / Reto; recuerda la última (si el navegador lo permite).
function pestanas(alCambiar) {
  const bs = [...document.querySelectorAll("nav.pestanas button")], vs = [...document.querySelectorAll(".vista")];
  const clave = "lab:" + location.pathname;
  const ir = (id) => {
    bs.forEach((b) => b.setAttribute("aria-selected", b.dataset.v === id));
    vs.forEach((v) => v.classList.toggle("activa", v.id === id));
    try { localStorage.setItem(clave, id); } catch (e) {}
    if (alCambiar) alCambiar(id);
  };
  bs.forEach((b) => b.addEventListener("click", () => ir(b.dataset.v)));
  let ini = bs[0].dataset.v; try { const g = localStorage.getItem(clave); if (g && document.getElementById(g)) ini = g; } catch (e) {}
  ir(ini);
}

// Grupo de botones de opción (uno a la vez).
function opciones(id, alElegir) {
  const bs = [...document.getElementById(id).querySelectorAll("button")];
  const elegir = (b) => { bs.forEach((x) => x.setAttribute("aria-pressed", x === b)); alElegir(b.dataset.v); };
  bs.forEach((b) => b.addEventListener("click", () => elegir(b)));
  return { elegir: (v) => elegir(bs.find((b) => b.dataset.v === String(v))) };
}

// Posición del puntero dentro del lienzo, en píxeles CSS.
function puntero(L, e) { const r = L.c.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
