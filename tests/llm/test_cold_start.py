from cf_pipeline.llm.cold_start import build_cold_start_prompt, parse_cold_start_response


def test_prompt_includes_history_titles():
    history = [{"title": "Toy Story", "genres": "Animation|Comedy"}]
    p = build_cold_start_prompt(user_id=42, history=history)
    assert "Toy Story" in p
    assert "JSON" in p


def test_prompt_no_history():
    p = build_cold_start_prompt(user_id=1, history=[])
    assert "(no history)" in p


def test_parse_valid_json():
    raw = '{"liked_genres": ["Action","Sci-Fi"], "liked_actors": ["Keanu Reeves"], "mood": "epic"}'
    parsed = parse_cold_start_response(raw)
    assert parsed["liked_genres"] == ["Action", "Sci-Fi"]
    assert parsed["mood"] == "epic"


def test_parse_handles_extra_text_around_json():
    raw = 'Sure! Here\'s the JSON: {"liked_genres":["X"],"liked_actors":[],"mood":"calm"} done.'
    parsed = parse_cold_start_response(raw)
    assert parsed["liked_genres"] == ["X"]


def test_parse_returns_default_on_invalid():
    parsed = parse_cold_start_response("nonsense")
    assert parsed == {"liked_genres": [], "liked_actors": [], "mood": ""}


def test_parse_returns_default_on_broken_json():
    parsed = parse_cold_start_response("{bad json here")
    assert parsed == {"liked_genres": [], "liked_actors": [], "mood": ""}


def test_parse_missing_keys_use_defaults():
    raw = '{"liked_genres": ["Drama"]}'
    parsed = parse_cold_start_response(raw)
    assert parsed["liked_genres"] == ["Drama"]
    assert parsed["liked_actors"] == []
    assert parsed["mood"] == ""
