/* Ukur latency tiap interaksi Dash (round-trip callback) & tampilkan di navbar.
   Defensif: semua dibungkus try/catch agar tak pernah mengganggu app. */
(function () {
  try {
    var orig = window.fetch;
    if (!orig) return;
    window.fetch = function () {
      var args = arguments, url = '';
      try { url = (args[0] && args[0].url) ? args[0].url : ('' + args[0]); } catch (e) {}
      if (url.indexOf('_dash-update-component') === -1) return orig.apply(this, args);
      var now = function () { return (window.performance && performance.now) ? performance.now() : Date.now(); };
      var t0 = now();
      var p = orig.apply(this, args);
      try {
        p.then(function (r) {
          try {
            var ms = Math.round(now() - t0);
            var el = document.getElementById('latency-badge');
            if (el) {
              el.textContent = '⚡ ' + ms + ' ms';
              var green = '#15803D', amber = '#B45309', red = '#DC2626';
              var col = ms < 200 ? green : (ms < 500 ? amber : red);
              var brd = ms < 200 ? '#BCE3D9' : (ms < 500 ? '#F3D9AE' : '#F4B4B4');
              el.style.color = col; el.style.borderColor = brd;
            }
          } catch (e) {}
          return r;
        });
      } catch (e) {}
      return p;
    };
  } catch (e) {}
})();
