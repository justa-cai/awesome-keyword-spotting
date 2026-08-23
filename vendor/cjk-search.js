/* CJK-friendly substring search for docsify (replaces built-in search plugin).
 * Serves 305 notes: lazy-builds a full-text index from sidebar links on first query,
 * supports Chinese queries and space-separated AND terms, highlights snippets. */
(function () {
  var INDEX = null, building = null, timer = null;

  function build() {
    if (INDEX) return Promise.resolve();
    if (building) return building;
    building = new Promise(function (resolve) {
      /* docsify rewrites sidebar anchor hrefs at render time, so parse the
       * _sidebar.md source instead — links there keep the #/papers/... form */
      fetch('/_sidebar.md').then(function (r) { return r.text(); }).then(function (md) {
        var seen = {}, paths = [], re = /\(#\/(papers\/[^)#\s]+)\)/g, m;
        while ((m = re.exec(md))) {
          if (!seen[m[1]]) { seen[m[1]] = 1; paths.push(m[1]); }
        }
        return Promise.all(paths.map(function (p) {
          return fetch(p + '.md').then(function (r) { return r.ok ? r.text() : ''; }).catch(function () { return ''; });
        })).then(function (txts) {
          INDEX = paths.map(function (p, i) {
            var t = txts[i];
            var mm = t.match(/^#\s+(.+)$/m);
            return { path: p, title: mm ? mm[1].trim() : p, text: t.toLowerCase() };
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
    var terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length || !INDEX) return [];
    var hits = [];
    for (var i = 0; i < INDEX.length; i++) {
      var d = INDEX[i];
      var ok = true, first = terms[0];
      for (var j = 0; j < terms.length; j++) {
        if (d.text.indexOf(terms[j]) < 0) { ok = false; break; }
        if (d.title.toLowerCase().indexOf(terms[j]) >= 0) d._t = true;
      }
      if (ok) hits.push(d);
      d._t = false;
    }
    hits.sort(function (a, b) { return (b._t ? 1 : 0) - (a._t ? 1 : 0); });
    return hits.slice(0, 30);
  }

  function mount(hook) {
    hook.doneEach(function () {
      var sidebar = document.querySelector('.sidebar');
      if (!sidebar || document.querySelector('.ksearch')) return;
      var box = document.createElement('div');
      box.className = 'ksearch';
      box.innerHTML = '<input placeholder="搜索论文 / 方法 / 关键词（支持中文）">'
        + '<div class="ksearch-results"></div>';
      sidebar.insertBefore(box, sidebar.firstChild);
      var inp = box.querySelector('input'), res = box.querySelector('.ksearch-results');
      inp.addEventListener('input', function () {
        clearTimeout(timer);
        var q = inp.value.trim();
        timer = setTimeout(function () {
          if (!q) { res.innerHTML = ''; return; }
          res.innerHTML = '<div class="ksearch-hint">' + (INDEX ? '搜索中…' : '首次搜索正在构建索引（305 篇，约几秒）…') + '</div>';
          build().then(function () {
            var hits = search(q);
            if (!hits.length) { res.innerHTML = '<div class="ksearch-hint">没有找到 “' + esc(q) + '”</div>'; return; }
            res.innerHTML = hits.map(function (d) {
              return '<a class="ksearch-item" href="#/' + d.path + '">'
                + '<div class="ksearch-title">' + esc(d.title) + '</div>'
                + '<div class="ksearch-snippet">' + snippet(d.text, q.toLowerCase().split(/\s+/)[0]) + '</div></a>';
            }).join('');
          });
        }, 250);
      });
    });
  }

  window.$docsify = window.$docsify || {};
  window.$docsify.plugins = (window.$docsify.plugins || []).concat([mount]);
})();
