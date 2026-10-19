import asyncio
import logging
from pathlib import Path

from conftest import TODAY, FakeLLM, raw_row

from healthtrack.config import OFFLINE_MODEL
from healthtrack.generator import FollowUpGenerator, parse_batch_response
from healthtrack.logging_config import setup_logging
from healthtrack.preprocessing import preprocess_records
from healthtrack.rag import tip_coverage


def patients(kb, n=5, **kw):
    rows = [raw_row(patient_id=f"P{i:03d}", **kw) for i in range(n)]
    return preprocess_records(rows, kb=kb, today=TODAY).records


def generate(kb, llm, recs, **kw):
    gen = FollowUpGenerator(kb, llm)
    return asyncio.run(gen.generate(recs, model=kw.pop("model", "mistral-small-latest"), **kw))


def test_one_message_per_patient_from_llm(kb):
    recs = patients(kb, 7)
    results = generate(kb, FakeLLM(), recs, batch_size=3)
    assert [r.patient_id for r in results] == [p.patient_id for p in recs]  # order preserved
    assert all(r.source == "llm" and r.status == "success" and r.message.startswith("Hi Rahul") for r in results)


def test_batching_reduces_llm_calls(kb):
    llm = FakeLLM()
    generate(kb, llm, patients(kb, 25), batch_size=10)
    assert llm.calls == 3


def test_every_message_contains_a_retrieved_health_tip_even_if_llm_forgets(kb):
    results = generate(kb, FakeLLM(include_tips=False), patients(kb, 6, condition="asthma"), batch_size=3)
    for r in results:
        assert r.tips and any(tip_coverage(r.message, t) >= 0.25 for t in r.tips)


def test_llm_failure_falls_back_to_template_for_everyone(kb):
    results = generate(kb, FakeLLM(fail=True), patients(kb, 4))
    assert len(results) == 4
    assert all(r.status == "fallback" and r.source == "template" and "429" in r.error and r.message for r in results)


def test_unparseable_llm_output_falls_back(kb):
    results = generate(kb, FakeLLM(garbage=True), patients(kb, 3))
    assert all(r.status == "fallback" for r in results)


def test_patients_omitted_by_llm_get_fallback_only_for_those(kb):
    results = generate(kb, FakeLLM(drop_every=2), patients(kb, 4), batch_size=4)
    assert [r.status for r in results] == ["fallback", "success", "fallback", "success"]


def test_offline_model_and_missing_llm_use_templates(kb):
    recs = patients(kb, 3)
    for llm, model in ((FakeLLM(), OFFLINE_MODEL), (None, "mistral-small-latest")):
        results = generate(kb, llm, recs, model=model)
        assert all(r.source == "template" and r.status == "success" for r in results)


def test_no_response_patients_get_reminder_and_escalation_wording(kb):
    recs = preprocess_records([
        raw_row(patient_id="A", response_status="no_response", attempts="1"),
        raw_row(patient_id="B", response_status="no_response", attempts="3"),
    ], kb=kb, today=TODAY).records
    a, b = generate(kb, None, recs, model=OFFLINE_MODEL)
    assert "not heard back" in a.message and "care team will call" in b.message


def test_parse_batch_response_variants():
    assert parse_batch_response('{"messages":[{"ref":"1","message":"hi"}]}') == {"1": "hi"}
    assert parse_batch_response('```json\n[{"ref":2,"message":"yo"}]\n```') == {"2": "yo"}
    for bad in ("nope", '{"x":1}'):
        try:
            parse_batch_response(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_log_file_records_messages_and_errors(kb, tmp_path: Path):
    log_path = setup_logging(tmp_path)
    generate(kb, FakeLLM(fail=True), patients(kb, 2))
    for h in logging.getLogger("healthtrack").handlers:
        h.flush()
    text = log_path.read_text()
    assert "message generated | patient=P000" in text and "ERROR" in text and "LLM batch failed" in text


def test_paraphrased_tip_is_not_duplicated(kb):
    class Para(FakeLLM):
        async def complete(self, *, model, messages, max_tokens):
            import json
            ps = json.loads(messages[-1]["content"])["patients"]
            return json.dumps({"messages": [{"ref": p["ref"], "message": f"Hi {p['first_name']}, thanks for visiting. Check BP daily and cut back on salt."} for p in ps]})
    rec = patients(kb, 1, condition="BP")
    r = generate(kb, Para(), rec)[0]
    assert "Tip:" not in r.message  # paraphrase already covers the tip


def test_missing_follow_up_date_and_cues_are_enforced(kb):
    recs = preprocess_records([
        raw_row(patient_id="A"),
        raw_row(patient_id="B", response_status="no_response", attempts="1"),
        raw_row(patient_id="C", response_status="no_response", attempts="3"),
    ], kb=kb, today=TODAY).records
    a, b, c = generate(kb, FakeLLM(), recs)
    for r, p in zip((a, b, c), recs, strict=True):
        assert "Follow-up due by" in r.message and p.follow_up_date.strftime("%b") in r.message
    assert "reply" in b.message.lower() and "care team will call" in c.message


def test_existing_date_mentions_are_recognised():
    from datetime import date

    from healthtrack.generator import mentions_date
    d = date(2026, 10, 23)
    assert all(mentions_date(t, d) for t in ("see you 23 Oct", "on 23rd October", "Oct 23!", "23 Oct 2026"))
    assert not mentions_date("see you on 3 Oct", d) and not mentions_date("no date", d)


def test_message_text_can_be_kept_out_of_logs(kb, tmp_path: Path):
    log_path = setup_logging(tmp_path)
    gen = FollowUpGenerator(kb, None, log_message_text=False)
    asyncio.run(gen.generate(patients(kb, 1), model=OFFLINE_MODEL))
    for h in logging.getLogger("healthtrack").handlers:
        h.flush()
    text = log_path.read_text()
    assert "message generated | patient=P000" in text and "text=" not in text and "chars=" in text
