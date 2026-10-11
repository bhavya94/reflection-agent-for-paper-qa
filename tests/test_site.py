import json

import pytest

from reflective_qa.site import TEMPLATE, render


@pytest.mark.parametrize("answer", ["</script><script>alert(1)</script>", "<!--<script>"])
def test_data_is_embedded_and_cannot_change_where_the_script_ends(answer):
    html = render({"answer": answer}, "<script>const DATA = __DATA__;</script>")
    embedded = html[len("<script>const DATA = "):-len(";</script>")]
    assert "<" not in embedded
    assert json.loads(embedded) == {"answer": answer}


def test_the_template_must_have_exactly_one_placeholder():
    with pytest.raises(ValueError):
        render({}, "no placeholder")
    assert TEMPLATE.read_text().count("__DATA__") == 1
