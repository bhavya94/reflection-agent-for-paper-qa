from reflective_qa.checks import Finding, is_yes_no_question, load_max_words, run_checks

IDS = {"s0p0", "s1p0"}


def _checks(question="What dataset is used?", answer="Foo", cited=("s0p0",), ids=IDS, max_words=5):
    return [f.check for f in run_checks(question, answer, cited, ids, max_words)]


def test_a_short_cited_answer_passes_every_check():
    assert _checks() == []


def test_citing_a_paragraph_that_was_not_given_is_flagged_with_its_id():
    (finding,) = run_checks("What dataset is used?", "Foo", ("s9p9",), IDS, 5)
    assert finding == Finding("bad_citation", "Cites s9p9, which were not among the given paragraphs.")


def test_an_answer_without_citations_is_flagged_except_closed_book():
    assert _checks(cited=()) == ["bad_citation"]
    assert _checks(cited=(), ids=set()) == []


def test_abstaining_while_citing_suggests_the_answer_is_there():
    assert _checks(answer="Unanswerable.") == ["wrongly_unanswerable"]
    assert _checks(answer="Unanswerable", cited=()) == []


def test_yes_no_questions_need_a_yes_or_no_answer():
    assert is_yes_no_question("Do they use BERT?") and not is_yes_no_question("Do they use BERT or ELMo?")
    assert _checks("Do they use BERT?", "They use BERT-base.") == ["yes_no_format"]
    assert _checks("Do they use BERT?", "yes.") == []


def test_long_answers_are_flagged_but_abstentions_are_not():
    assert _checks(answer="one two three four five six") == ["too_long"]
    assert _checks(answer="Unanswerable", cited=(), max_words=0) == []


def test_the_word_limit_comes_from_config():
    assert load_max_words() == 30
