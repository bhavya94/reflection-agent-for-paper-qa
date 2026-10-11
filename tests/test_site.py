import json

import pytest

from reflective_qa.site import TEMPLATE, render


def test_data_is_embedded_and_cannot_close_the_script_tag():
    html = render({"answer": "</script><script>alert(1)</script>"}, "<script>const DATA = __DATA__;</script>")
    assert "</script><script>alert" not in html
    embedded = html[len("<script>const DATA = "):-len(";</script>")]
    assert json.loads(embedded.replace("<\\/", "</")) == {"answer": "</script><script>alert(1)</script>"}


def test_the_template_must_have_exactly_one_placeholder():
    with pytest.raises(ValueError):
        render({}, "no placeholder")
    assert TEMPLATE.read_text().count("__DATA__") == 1
