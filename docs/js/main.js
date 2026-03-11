// ── TAB SWITCHING ─────────────────────────────────────────────────────────────
function showTab(tabId, btnEl) {
  // Hide all panels and deactivate all buttons
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));

  document.getElementById(tabId).classList.add('active');
  btnEl.classList.add('active');
}

// ── SCROLL REVEAL ─────────────────────────────────────────────────────────────
function initReveal() {
  const targets = document.querySelectorAll(
    '.section-inner > *, .stat-card, .viz-img, .data-finding, .emphasis-block, .argument-list'
  );
  targets.forEach(el => el.classList.add('reveal'));

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.add('visible');
          observer.unobserve(e.target);
        }
      });
    },
    { threshold: 0.08, rootMargin: '0px 0px -40px 0px' }
  );

  targets.forEach(el => observer.observe(el));
}

// ── NAV ACTIVE SECTION HIGHLIGHT ─────────────────────────────────────────────
function initNavHighlight() {
  const sections = document.querySelectorAll('section[id], div[id]');
  const navLinks = document.querySelectorAll('.nav-links a');

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          navLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href === `#${id}`) {
              link.style.color = 'var(--green)';
            } else {
              link.style.color = '';
            }
          });
        }
      });
    },
    { threshold: 0.35 }
  );

  sections.forEach(s => observer.observe(s));
}

// ── SPIRAL CANVAS HERO ANIMATION ──────────────────────────────────────────────
function drawHeroSpiral() {
  const hero = document.getElementById('hero');
  if (!hero) return;

  const canvas = document.createElement('canvas');
  canvas.style.cssText = `
    position: absolute; inset: 0; width: 100%; height: 100%;
    pointer-events: none; opacity: 0.18; z-index: 0;
  `;
  hero.insertBefore(canvas, hero.firstChild);

  const ctx = canvas.getContext('2d');
  let w, h, t = 0;

  function resize() {
    w = canvas.width  = hero.offsetWidth;
    h = canvas.height = hero.offsetHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  // Fibonacci / golden spiral paths — multiple arms
  function drawSpiral(cx, cy, scale, hue, alpha, tOffset) {
    ctx.beginPath();
    const phi = 1.618033988;
    let r = 0;
    let first = true;
    for (let theta = 0 + tOffset; theta < Math.PI * 20; theta += 0.05) {
      r = scale * Math.pow(phi, theta / (Math.PI * 2));
      const x = cx + r * Math.cos(theta);
      const y = cy + r * Math.sin(theta);
      if (first) { ctx.moveTo(x, y); first = false; }
      else ctx.lineTo(x, y);
      if (r > Math.max(w, h)) break;
    }
    ctx.strokeStyle = `hsla(${hue}, 80%, 60%, ${alpha})`;
    ctx.lineWidth = 0.8;
    ctx.stroke();
  }

  let animId;
  function render() {
    ctx.clearRect(0, 0, w, h);
    // Slow drift
    const drift = t * 0.0003;

    drawSpiral(w * 0.65, h * 0.5, 2,  152, 0.35, drift);       // green spiral
    drawSpiral(w * 0.35, h * 0.55, 1.5, 45, 0.18, -drift + 1); // gold ghost spiral
    drawSpiral(w * 0.72, h * 0.4, 0.8, 152, 0.15, drift * 2);  // small spiral

    t++;
    animId = requestAnimationFrame(render);
  }
  render();
}

// ── INIT ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initReveal();
  initNavHighlight();
  drawHeroSpiral();
});
