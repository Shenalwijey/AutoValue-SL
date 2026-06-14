/**
 * script.js – AutoValue SL
 */

// Navbar scroll effect
(function () {
  const nav = document.getElementById("mainNav");
  if (!nav) return;
  window.addEventListener("scroll", () => {
    nav.style.padding = window.scrollY > 40 ? "0.4rem 0" : "0.75rem 0";
    nav.style.boxShadow = window.scrollY > 40 ? "0 4px 30px rgba(0,0,0,0.3)" : "none";
  });
})();

// Smooth scroll
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener("click", function(e) {
    const t = document.querySelector(this.getAttribute("href"));
    if (t) { e.preventDefault(); t.scrollIntoView({ behavior: "smooth" }); }
  });
});

// Form validation
(function () {
  const form = document.getElementById("predictForm");
  if (!form) return;
  const currentYear = new Date().getFullYear();
  form.addEventListener("submit", function(e) {
    let valid = true;
    form.querySelectorAll("[required]").forEach(f => {
      f.classList.remove("is-invalid");
      if (!f.value.trim()) { f.classList.add("is-invalid"); valid = false; }
    });
    const year = form.querySelector('[name="year"]');
    if (year && (year.value < 1990 || year.value > currentYear)) { year.classList.add("is-invalid"); valid = false; }
    if (!valid) {
      e.preventDefault();
      const first = form.querySelector(".is-invalid");
      if (first) first.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  });
  form.querySelectorAll(".form-control,.form-select").forEach(f =>
    f.addEventListener("input", () => f.classList.remove("is-invalid"))
  );
})();

// Animate hero stat numbers
(function () {
  const stats = document.querySelectorAll(".h-stat-num");
  if (!stats.length) return;
  const obs = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const el = entry.target;
      const raw = parseFloat(el.textContent.replace(/[^0-9.]/g, ""));
      const suffix = el.textContent.replace(/[0-9.,]/g, "");
      if (isNaN(raw)) return;
      const start = performance.now();
      const run = now => {
        const p = Math.min((now - start) / 900, 1);
        const ease = 1 - Math.pow(1 - p, 3);
        el.textContent = (Number.isInteger(raw) ? Math.round(raw * ease).toLocaleString() : (raw * ease).toFixed(1)) + suffix;
        if (p < 1) requestAnimationFrame(run);
      };
      requestAnimationFrame(run);
      obs.unobserve(el);
    });
  }, { threshold: 0.5 });
  stats.forEach(el => obs.observe(el));
})();
