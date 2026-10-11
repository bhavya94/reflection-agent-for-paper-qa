from reflective_qa.checks import Finding, is_yes_no_question, load_max_words, run_checks

IDS = {"s0p0", "s1p0"}


def _checks(question="What dataset is used?", answer="Foo", cited=("s0p0",), ids=IDS, max_words=5):
    return [f.check for f in run_checks(question, answer, cited, ids, max_words)]


def test_a_short_cited_answer_passes_every_check():
    assert _checks() == []


def test_citing_a_paragraph_that_was_not_given_is_flagged_once_per_id():
    (finding,) = run_checks("What dataset is used?", "Foo", ("s9p9", "s9p9"), IDS, 5)
    assert finding == Finding("bad_citation", "Cites s9p9, which was not among the given paragraphs.")


def test_citations_are_not_checked_closed_book():
    assert _checks(cited=()) == ["bad_citation"]
    assert _checks(cited=(), ids=set()) == _checks(cited=("s9p9",), ids=set()) == []


def test_an_abstention_should_cite_nothing():
    assert _checks(answer="Unanswerable.") == ["bad_citation"]
    assert _checks(answer="Unanswerable", cited=()) == []


def test_an_empty_answer_gets_no_findings():
    assert _checks("Do they use BERT?", answer="  ", cited=()) == []


def test_yes_no_questions_need_a_yes_or_no_answer():
    assert is_yes_no_question("  Do they use BERT?") and not is_yes_no_question("Do they use BERT or ELMo?")
    assert not is_yes_no_question("Domain adaptation: which method?")
    assert _checks("Do they use BERT?", "They use BERT-base.") == ["yes_no_format"]
    assert _checks("Do they use BERT?", "yes.") == _checks("Do they use BERT?", "Unanswerable", cited=()) == []


def test_long_answers_are_flagged_but_abstentions_are_not():
    assert _checks(answer="one two three four five six") == ["too_long"]
    assert _checks(answer="one two three four five") == []  # exactly the limit
    assert _checks(answer="Unanswerable", cited=(), max_words=0) == []


def test_the_word_limit_comes_from_config():
    assert load_max_words() == 35
