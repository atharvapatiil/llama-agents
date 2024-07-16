import pytest
from pytest import raises
from unittest.mock import Mock, MagicMock

from llama_agents import ServiceComponent, QueueMessage
from llama_agents.orchestrators.pipeline import get_service_component_message
from llama_agents.tools.service_component import ModuleType
from llama_agents.types import ActionTypes


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
