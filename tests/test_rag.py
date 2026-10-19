import pytest

from healthtrack.rag import tip_coverage


@pytest.mark.parametrize("key,word", [("diabetes", "sugar|insulin|diabetes"), ("asthma", "inhaler"),
                                      ("hypertension", "blood pressure"), ("back_pain", "stretch|lifting|back")])
def test_retrieves_tips_relevant_to_condition(kb, key, word):
    tips = kb.retrieve(key, top_k=2)
    assert len(tips) == 2 and all(t.condition == key for t in tips)
    assert any(any(w in t.text.lower() for w in word.split("|")) for t in tips)


def test_notes_steer_ranking_within_condition(kb):
    top = kb.retrieve("diabetes", "missed insulin dose", top_k=1)[0]
    assert "insulin" in top.text.lower()
    top = kb.retrieve("diabetes", "foot wound not healing", top_k=1)[0]
    assert "feet" in top.text.lower()


def test_unknown_condition_and_empty_query_still_return_a_tip(kb):
    assert kb.retrieve("does-not-exist", "", top_k=2)
    assert kb.retrieve("general", "")


def test_small_conditions_are_padded_with_general_tips(kb):
    assert len(kb.retrieve("uti", top_k=5)) == 5


def test_every_condition_has_tips_and_resolves_from_its_label(kb):
    for key in kb.conditions:
        assert kb.retrieve(key)
        assert kb.resolve(kb.label(key)) == key


def test_longest_alias_wins(kb):
    assert kb.resolve("knee injury after fall") == "knee_injury"
    assert kb.resolve("type 2 diabetes") == "diabetes"


def test_tip_coverage():
    tip = "Keep your reliever inhaler with you at all times."
    assert tip_coverage("Remember to keep your reliever inhaler with you.", tip) >= 0.6
    assert tip_coverage("Thanks for visiting us.", tip) < 0.2
