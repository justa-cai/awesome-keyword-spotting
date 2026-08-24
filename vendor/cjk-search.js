/* Bilingual (zh/en) search + EN sidebar link rewrite for docsify.
 * - Search indexes the current language's notes (EN falls back to ZH content
 *   for untranslated papers, linking straight to the ZH route).
 * - After sidebar render, EN routes rewrite #/papers/... hrefs to #/en/papers/...
 *   for translated papers only (manifest-driven); untranslated ones stay ZH. */
(function () {
  var CACHE = {}; // lang -> index array
  var building = null, timer = null, lastLang = null;
  var EN_SET = null, EN_LOADING = null;

  function lang() { return /^#\/en\//.test(location.hash) ? 'en' : 'zh'; }

  function enSet() {
    if (EN_SET) return Promise.resolve(EN_SET);
    if (EN_LOADING) return EN_LOADING;
    EN_LOADING = fetch('/en/manifest.json').then(function (r) {
      return r.ok ? r.json() : [];
    }).then(function (list) {
      EN_SET = {};
      (list || []).forEach(function (p) { EN_SET[p] = 1; });
      return EN_SET;
    }).catch(function () { EN_SET = {}; return EN_SET; });
    return EN_LOADING;
  }

  /* sidebar href rewrite: EN route → point translated papers at their EN pages */
  function rewriteSidebar() {
    if (lang() !== 'en') return;
    enSet().then(function (set) {
      [].slice.call(document.querySelectorAll('.sidebar a[href^="#/papers/"]')).forEach(function (a) {
        var p = a.getAttribute('href').slice(2); // papers/...
        if (set[p]) a.setAttribute('href', '#/en/' + p);
      });
    });
  }

  function build() {
    var L = lang();
    if (CACHE[L]) return Promise.resolve();
    if (building && building._lang === L) return building;
    building = new Promise(function (resolve) {
      building._lang = L;
      var L2 = L;
      var sb = L2 === 'en' ? '/en/_sidebar.md' : '/_sidebar.md';
      fetch(sb).then(function (r) { return r.text(); }).then(function (md) {
        var seen = {}, paths = [], re = /\((?!#)(papers\/[^)#\s]+)\)/g, m;
        while ((m = re.exec(md))) {
          if (!seen[m[1]]) { seen[m[1]] = 1; paths.push(m[1]); }
        }
        var fetchOne = function (p) {
          if (L2 === 'zh') {
            return fetch(p + '.md').then(function (r) { return r.ok ? r.text() : ''; }).catch(function () { return ''; })
              .then(function (t) { return { text: t, route: '#/' + p }; });
          }
          return fetch('/en/' + p + '.md').then(function (r) {
            if (r.ok) return r.text().then(function (t) { return { text: t, route: '#/en/' + p }; });
            return fetch(p + '.md').then(function (r2) { return r2.ok ? r2.text() : ''; }).catch(function () { return ''; })
              .then(function (t) { return { text: t, route: '#/' + p }; });
          }).catch(function () { return { text: '', route: '#/' + p }; });
        };
        Promise.all(paths.map(fetchOne)).then(function (docs) {
          CACHE[L2] = paths.map(function (p, i) {
            var t = docs[i].text;
            var mm = t.match(/^#\s+(.+)$/m);
            return { route: docs[i].route, title: mm ? mm[1].trim() : p, text: t.toLowerCase() };
          });
          resolve();
        });
      });
    });
    return building;
  }

  function esc(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function snippet(text, q) {
    var i = text.indexOf(q);
    if (i < 0) return '';
    var s = Math.max(0, i - 40);
    var out = text.slice(s, i + q.length + 60).replace(/\s+/g, ' ').trim();
    return esc(out).replace(esc(q), '<mark>' + esc(q) + '</mark>');
  }

  function search(q) {
    var idx = CACHE[lang()] || [];
    var terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) return [];
    var hits = [];
    for (var i = 0; i < idx.length; i++) {
      var d = idx[i], ok = true, inTitle = false;
      for (var j = 0; j < terms.length; j++) {
        if (d.text.indexOf(terms[j]) < 0) { ok = false; break; }
        if (d.title.toLowerCase().indexOf(terms[j]) >= 0) inTitle = true;
      }
      if (ok) { d._t = inTitle; hits.push(d); }
    }
    hits.sort(function (a, b) { return (b._t ? 1 : 0) - (a._t ? 1 : 0); });
    return hits.slice(0, 30);
  }

  function mount(hook) {
    hook.doneEach(function () {
      var sidebar = document.querySelector('.sidebar');
      if (!sidebar) return;
      rewriteSidebar();

      var L = lang();
      var box = document.querySelector('.ksearch');
      if (!box) {
        box = document.createElement('div');
        box.className = 'ksearch';
        box.innerHTML = '<input><div class="ksearch-results"></div>';
        sidebar.insertBefore(box, sidebar.firstChild);
        box.querySelector('input').addEventListener('input', function () {
          clearTimeout(timer);
          var q = this.value.trim();
          var res = box.querySelector('.ksearch-results');
          timer = setTimeout(function () { run(q, res); }, 250);
        });
      }
      var inp = box.querySelector('input'), res = box.querySelector('.ksearch-results');
      inp.placeholder = L === 'en'
        ? 'Search papers / methods / keywords'
        : '搜索论文 / 方法 / 关键词（支持中文）';
      if (lastLang !== L) { lastLang = L; res.innerHTML = ''; inp.value = ''; }

      function run(q, res) {
        if (!q) { res.innerHTML = ''; return; }
        var total = (CACHE[lang()] || []).length || 305;
        res.innerHTML = '<div class="ksearch-hint">' + (CACHE[lang()]
          ? 'Searching…'
          : 'Building index (' + total + ' notes, a few seconds)…') + '</div>';
        build().then(function () {
          var hits = search(q);
          if (!hits.length) { res.innerHTML = '<div class="ksearch-hint">No results for “' + esc(q) + '”</div>'; return; }
          res.innerHTML = hits.map(function (d) {
            return '<a class="ksearch-item" href="' + d.route + '">'
              + '<div class="ksearch-title">' + esc(d.title) + '</div>'
              + '<div class="ksearch-snippet">' + snippet(d.text, q.toLowerCase().split(/\s+/)[0]) + '</div></a>';
          }).join('');
        });
      }
    });
  }

  window.$docsify = window.$docsify || {};
  window.$docsify.plugins = (window.$docsify.plugins || []).concat([mount]);
})();
