"""The dashboard's static assets are served (tornado x bokeh compatibility).

Panel serves its JavaScript through bokeh's ``MultiRootStaticHandler``. With
tornado 6.5.9 that handler raises on every request (it overrides
``initialize`` without setting the ``allowed_symlink_directory`` attribute the
new tornado reads), so ``biosim dashboard`` and ``docker compose up`` served a
blank page while every Python-level test still passed. This drives the real
handler through a real HTTP request.
"""

from __future__ import annotations

import pytest

pytest.importorskip("bokeh")
tornado_testing = pytest.importorskip("tornado.testing")

from bokeh.server.views.multi_root_static_handler import MultiRootStaticHandler  # noqa: E402
from tornado.web import Application  # noqa: E402


@pytest.fixture()
def asset_dir(tmp_path):
    (tmp_path / "panel.min.js").write_text("/* ok */")
    return tmp_path


def test_bokeh_static_handler_serves_a_file(asset_dir):
    class _Case(tornado_testing.AsyncHTTPTestCase):
        def get_app(self):
            return Application([(r"/static/extensions/(.*)", MultiRootStaticHandler,
                                 {"root": {"panel": str(asset_dir)}})])

        def runTest(self):  # noqa: N802 - unittest API
            response = self.fetch("/static/extensions/panel/panel.min.js")
            assert response.code == 200, (
                f"static asset returned {response.code}: the installed tornado and bokeh "
                "are incompatible (see the tornado pin in pyproject.toml)"
            )

    case = _Case()
    case.setUp()
    try:
        case.runTest()
    finally:
        case.tearDown()
