from pathlib import Path

from src.chat_service import FALLBACK_ANSWER, answer_question


def _write_policy_files(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "leave_policy.txt").write_text("Leave policy covers casual and sick leave.", encoding="utf-8")
    (folder / "travel_policy.txt").write_text("Travel policy covers hotel and cab reimbursement.", encoding="utf-8")
    (folder / "it_support_faq.txt").write_text("IT support covers laptop and password helpdesk issues.", encoding="utf-8")


def test_leave_keyword_maps_to_leave_policy(tmp_path: Path):
    _write_policy_files(tmp_path)

    response = answer_question("How does sick leave work?", raw_dir=tmp_path)

    assert response["source_file"] == "leave_policy.txt"


def test_travel_keyword_maps_to_travel_policy(tmp_path: Path):
    _write_policy_files(tmp_path)

    response = answer_question("Can I claim hotel reimbursement?", raw_dir=tmp_path)

    assert response["source_file"] == "travel_policy.txt"


def test_laptop_password_keyword_maps_to_it_policy(tmp_path: Path):
    _write_policy_files(tmp_path)

    laptop_response = answer_question("I lost my laptop.", raw_dir=tmp_path)
    password_response = answer_question("How do I reset my password?", raw_dir=tmp_path)

    assert laptop_response["source_file"] == "it_support_faq.txt"
    assert password_response["source_file"] == "it_support_faq.txt"


def test_unknown_question_returns_fallback_response(tmp_path: Path):
    _write_policy_files(tmp_path)

    response = answer_question("Where is the lunch menu?", raw_dir=tmp_path)

    assert response["answer"] == FALLBACK_ANSWER
    assert response["source_file"] is None


def test_response_identifies_keyword_lookup_and_not_llm(tmp_path: Path):
    _write_policy_files(tmp_path)

    response = answer_question("Tell me about travel.", raw_dir=tmp_path)

    assert response["retrieval_method"] == "keyword_lookup"
    assert response["is_llm_response"] is False
