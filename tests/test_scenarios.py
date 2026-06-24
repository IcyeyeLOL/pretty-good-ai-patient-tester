from __future__ import annotations

from scenarios.scenario_cards import ScenarioCard, get_all_scenarios, get_scenario


def test_all_scenarios_are_present_and_ordered():
    scenarios = get_all_scenarios()

    assert len(scenarios) == 10
    assert [scenario.id for scenario in scenarios] == list(range(1, 11))


def test_every_scenario_card_has_required_fields():
    for scenario in get_all_scenarios():
        assert isinstance(scenario, ScenarioCard)
        assert scenario.name
        assert scenario.caller_name
        assert scenario.goal
        assert scenario.expected_agent_behavior
        assert scenario.bug_conditions
        assert scenario.severity in {"low", "medium", "high"}
        assert scenario.end_condition
        assert scenario.opening_line


def test_get_scenario_returns_expected_card():
    scenario = get_scenario(1)

    assert scenario is not None
    assert scenario.caller_name == "Maria Johnson"
    assert "new patient" in scenario.opening_line.lower()


def test_get_unknown_scenario_returns_none():
    assert get_scenario(99) is None


def test_high_risk_scenarios_are_marked_high_severity():
    high_risk_ids = {2, 4, 5, 6}

    for scenario_id in high_risk_ids:
        assert get_scenario(scenario_id).severity == "high"  # type: ignore[union-attr]
