from typing import Set, Any, Dict

import pytest
from llama_index.core.base.query_pipeline.query import QueryComponent
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.query_pipeline.query import RunState
from pytest import raises
from unittest.mock import Mock, MagicMock, create_autospec

from llama_agents import ServiceComponent, QueueMessage
from llama_agents.orchestrators.pipeline import (
    get_service_component_message,
    process_component_output,
)
from llama_agents.tools.service_component import ModuleType
from llama_agents.types import ActionTypes, TaskResult
from pydantic import Field


@pytest.fixture
def agent_service_component() -> ServiceComponent:
    """Fixture to provide an agent service component setup."""
    return ServiceComponent(
        name="AgentService",
        description="Agent Service Description",
        module_type=ModuleType.AGENT,
    )


@pytest.fixture
def component_service_component() -> ServiceComponent:
    """Fixture to provide a component service component setup."""
    return ServiceComponent(
        name="ComponentService",
        description="Component Service Description",
        module_type=ModuleType.COMPONENT,
    )


def test_get_service_component_message_with_agent(
    mocker: MagicMock, agent_service_component: ServiceComponent
) -> None:
    """Test service component message creation for agent type components."""
    task_id: str = "task123"
    input_dict: dict = {"key": "value"}

    # Mock TaskDefinition using the actual module path
    mock_task_definition: Mock = mocker.patch(
        "llama_agents.types.TaskDefinition", autospec=True
    )
    mock_task_definition.return_value.model_dump.return_value = {
        "agent_id": None,
        "input": "value",
        "state": {},
        "task_id": task_id,
    }

    # Call the function
    result: QueueMessage = get_service_component_message(
        agent_service_component, task_id, input_dict
    )

    # Assertions
    expected_data: dict = {
        "agent_id": None,
        "input": "value",
        "state": {},
        "task_id": task_id,
    }
    assert isinstance(result, QueueMessage)
    assert result.type == "AgentService"
    assert result.action == ActionTypes.NEW_TASK
    assert result.data == expected_data


def test_get_service_component_message_with_component(
    mocker: MagicMock, component_service_component: ServiceComponent
) -> None:
    """Test service component message creation for component type components."""
    task_id: str = "task123"
    input_dict: dict = {"key1": "value1", "key2": "value2"}

    # Use mocker to patch TaskDefinition methods accurately with autospec
    mock_task_definition: Mock = mocker.patch(
        "llama_agents.types.TaskDefinition", autospec=True
    )
    mock_task_definition.return_value.model_dump.return_value = {
        "input": "",
        "task_id": task_id,
        "state": {"__input_dict__": input_dict},
        "agent_id": None,
    }

    # Call the function
    result: QueueMessage = get_service_component_message(
        component_service_component, task_id, input_dict
    )

    # Assertions
    expected_data: dict = {
        "input": "",
        "task_id": task_id,
        "state": {"__input_dict__": input_dict},
        "agent_id": None,
    }
    assert isinstance(result, QueueMessage)
    assert result.type == "ComponentService"
    assert result.action == ActionTypes.NEW_TASK
    assert result.data == expected_data


def test_get_service_component_message_with_invalid_module_type(
    mocker: MagicMock, agent_service_component: ServiceComponent
) -> None:
    """Test error handling for invalid module types by dynamically setting an unsupported enum value."""
    # Dynamically set an invalid module_type
    object.__setattr__(
        agent_service_component, "module_type", "INVALID"
    )  # Bypass enum type-checking

    task_id: str = "task123"
    input_dict: dict = {"key": "value"}

    mocker.patch("llama_agents.types.TaskDefinition", autospec=True)

    # Expecting a ValueError
    with raises(ValueError):
        get_service_component_message(agent_service_component, task_id, input_dict)


# Mock of QueryComponent to be used in tests
class ConcreteQueryComponent(QueryComponent):
    module_type: ModuleType = Field(
        default=None
    )  # Include module_type with Pydantic support

    def _arun_component(self) -> None:
        pass

    def _run_component(self) -> None:
        pass

    def _validate_component_inputs(self, _input: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def input_keys(self) -> Set[str]:
        return set()

    def output_keys(self) -> Set[str]:
        return set()

    def set_callback_manager(self, manager: Any) -> None:
        pass


@pytest.fixture
def pipeline() -> QueryPipeline:
    """Fixture to create a mocked QueryPipeline."""
    return create_autospec(QueryPipeline)


@pytest.fixture
def run_state() -> RunState:
    """Fixture to create a RunState with a default module."""
    module = ConcreteQueryComponent()
    module_dict = {"module1": module}
    module_input_dict = {"module1": {"input": "test"}}
    rs = RunState(module_dict=module_dict, module_input_dict=module_input_dict)
    rs.module_dict[
        "module1"
    ].module_type = ModuleType.AGENT  # Default to AGENT for general tests
    return rs


@pytest.fixture
def task_result() -> TaskResult:
    """Fixture to create a TaskResult."""
    return TaskResult(task_id="1", history=[], result="output", data={"key": "value"})


# Test function to handle different module types
def test_process_component_output(
    pipeline: QueryPipeline, run_state: RunState, task_result: TaskResult
) -> None:
    # Test with ModuleType.AGENT
    process_component_output(pipeline, run_state, "module1", task_result)
    pipeline.process_component_output.assert_called_once_with(
        {"output": task_result.result}, "module1", run_state
    )

    # Reset mock for the next call
    pipeline.process_component_output.reset_mock()

    # Test with ModuleType.COMPONENT
    run_state.module_dict["module1"].module_type = ModuleType.COMPONENT
    process_component_output(pipeline, run_state, "module1", task_result)
    pipeline.process_component_output.assert_called_once_with(
        task_result.data, "module1", run_state
    )

    # Reset mock and test for invalid module type
    pipeline.process_component_output.reset_mock()
    run_state.module_dict[
        "module1"
    ].module_type = "INVALID"  # Intentionally incorrect for testing
    with pytest.raises(ValueError):
        process_component_output(pipeline, run_state, "module1", task_result)

    # Test without setting module_type
    pipeline.process_component_output.reset_mock()
    delattr(
        run_state.module_dict["module1"], "module_type"
    )  # Remove attribute to simulate absence
    process_component_output(pipeline, run_state, "module1", task_result)
    pipeline.process_component_output.assert_called_once_with(
        {"output": task_result.result}, "module1", run_state
    )
