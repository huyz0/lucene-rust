// Wrap every table in a block that scrolls sideways only when the table is wider than the page.
(function () {
    function wrap() {
        document.querySelectorAll('.content main table').forEach(function (t) {
            if (t.parentElement && t.parentElement.classList.contains('table-wrap')) return;
            var d = document.createElement('div');
            d.className = 'table-wrap';
            t.parentNode.insertBefore(d, t);
            d.appendChild(t);
        });
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', wrap);
    else wrap();
})();
