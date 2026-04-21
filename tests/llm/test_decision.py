from cf_pipeline.llm.decision import build_decision_prompt, parse_decision_response


def test_prompt_has_strict_schema():
    p = build_decision_prompt(
        user_history=[{"title": "Toy Story", "genres": "Animation"}],
        retrieved=[{"title": "Cars", "genres": "Animation"}],
        candidate={"title": "Saw", "genres": "Horror", "overview": "trap"},
    )
    assert "YES" in p and "NO" in p
    assert "JSON" in p
    assert "Saw" in p
    assert "Toy Story" in p


def test_parse_yes():
    out = parse_decision_response('{"decision":"YES"}', None)
    assert out["decision"] == "YES"
    assert out["yes_prob"] == 1.0


def test_parse_no():
    out = parse_decision_response('{"decision":"NO"}', None)
    assert out["decision"] == "NO"
    assert out["yes_prob"] == 0.0


def test_parse_invalid_defaults_no():
    out = parse_decision_response("no opinion", None)
    assert out["decision"] == "NO"


def test_parse_extra_text_around_json():
    out = parse_decision_response('Sure! {"decision": "YES"} thanks', None)
    assert out["decision"] == "YES"


def test_parse_lowercase_yes():
    out = parse_decision_response('{"decision":"yes"}', None)
    assert out["decision"] == "YES"
