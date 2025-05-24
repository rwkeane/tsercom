import pytest
import datetime
from unittest.mock import MagicMock

from tsercom.data.exposed_data_with_responder import ExposedDataWithResponder
from tsercom.data.remote_data_responder import RemoteDataResponder
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.exposed_data import ExposedData


# --- Concrete implementation of RemoteDataResponder for testing ---
class ConcreteImplRemoteDataResponder(RemoteDataResponder):
    """A concrete implementation of RemoteDataResponder for testing purposes."""
    def __init__(self):
        # Mock the method that will be called, so we can make assertions on it.
        self._on_response_ready_mock = MagicMock()

    def _on_response_ready(self, response: any) -> None:
        self._on_response_ready_mock(response)

# --- Fixtures ---
@pytest.fixture
def mock_caller_id(mocker):
    """Provides a mock CallerIdentifier."""
    return MagicMock(spec=CallerIdentifier)

@pytest.fixture
def mock_timestamp(mocker):
    """Provides a mock datetime.datetime object."""
    return MagicMock(spec=datetime.datetime)

@pytest.fixture
def valid_responder_mock_method():
    """
    Provides an instance of ConcreteImplRemoteDataResponder.
    The instance itself is real, but its _on_response_ready_mock attribute is a MagicMock.
    """
    return ConcreteImplRemoteDataResponder()


def test_exposed_data_with_responder_initialization_success(
    mock_caller_id, mock_timestamp, valid_responder_mock_method
):
    """Tests successful initialization of ExposedDataWithResponder."""
    responder_instance = valid_responder_mock_method
    
    exposed_data = ExposedDataWithResponder(
        caller_id=mock_caller_id,
        timestamp=mock_timestamp,
        responder=responder_instance
    )
    assert exposed_data.caller_id is mock_caller_id
    assert exposed_data.timestamp is mock_timestamp
    # Accessing private __responder attribute for verification is common in testing
    assert exposed_data._ExposedDataWithResponder__responder is responder_instance


def test_exposed_data_with_responder_init_raises_assertion_error_if_responder_is_none(
    mock_caller_id, mock_timestamp
):
    """Tests that AssertionError is raised if the responder is None."""
    # The original assert is `assert responder is not None` (no custom message)
    with pytest.raises(AssertionError): # Removed 'match' as assert has no message
        ExposedDataWithResponder(
            caller_id=mock_caller_id,
            timestamp=mock_timestamp,
            responder=None
        )

def test_exposed_data_with_responder_init_raises_assertion_error_if_responder_is_invalid_type(
    mock_caller_id, mock_timestamp
):
    """Tests that AssertionError is raised if the responder is not a RemoteDataResponder subclass."""
    class NotAResponder: # Does not inherit from RemoteDataResponder
        pass

    invalid_responder = NotAResponder()
    # The original assert is `assert issubclass(type(responder), RemoteDataResponder)` (no custom message)
    with pytest.raises(AssertionError): # Removed 'match'
        ExposedDataWithResponder(
            caller_id=mock_caller_id,
            timestamp=mock_timestamp,
            responder=invalid_responder
        )

def test_exposed_data_with_responder_respond_method_calls_responder_on_response_ready(
    mock_caller_id, mock_timestamp, valid_responder_mock_method
):
    """Tests that _respond() calls _on_response_ready() on the responder."""
    responder_instance = valid_responder_mock_method
    
    exposed_data = ExposedDataWithResponder(
        caller_id=mock_caller_id,
        timestamp=mock_timestamp,
        responder=responder_instance
    )
    
    response_payload = {"data": "test_response"}
    exposed_data._respond(response_payload)
    
    # Assert that the mock method on our concrete responder instance was called
    responder_instance._on_response_ready_mock.assert_called_once_with(response_payload)

# Test with a direct mock that uses create_autospec for more robust type checking if needed
# This is an alternative to the ConcreteImplRemoteDataResponder approach
@pytest.fixture
def autospec_mock_responder(mocker):
    # Create a mock that automatically adheres to the spec of RemoteDataResponder
    # This includes making it pass isinstance checks if RemoteDataResponder is a registered ABC.
    # For issubclass(type(responder), RemoteDataResponder), this won't work directly as type(mock) is MagicMock.
    # So, the ConcreteImplRemoteDataResponder is generally better for the issubclass check.
    # However, create_autospec is good for ensuring method signatures are matched.
    mock = mocker.create_autospec(RemoteDataResponder, instance=True)
    return mock

def test_exposed_data_with_responder_init_with_autospec_mock_responder_fails_issubclass(
    mock_caller_id, mock_timestamp, autospec_mock_responder
):
    """
    Tests that create_autospec mock fails the issubclass(type(mock), ...) check,
    as type(mock) is MagicMock, not a subclass.
    """
    # This test verifies our understanding of why the previous generic MagicMock failed.
    with pytest.raises(AssertionError):
        ExposedDataWithResponder(
            caller_id=mock_caller_id,
            timestamp=mock_timestamp,
            responder=autospec_mock_responder # type(autospec_mock_responder) is MagicMock
        )

# The test 'test_exposed_data_with_responder_init_with_mocked_subclass_of_responder'
# from the previous attempt is now effectively covered by using ConcreteImplRemoteDataResponder
# in the `valid_responder_mock_method` fixture for successful cases.
# The `ConcreteImplRemoteDataResponder` serves as the "mocked subclass" that correctly implements abstract methods.
# The `autospec_mock_responder` test above clarifies why simple mocks fail the `issubclass` check.
# No need for a separate `MockedSubResponder` class in the tests.
# The `test_exposed_data_with_responder_init_with_concrete_responder` is also covered by the success test.
# The key is that the `responder` argument must be an instance of a class that *is a subclass* of RemoteDataResponder.
# `ConcreteImplRemoteDataResponder` fulfills this.
# The `valid_responder_mock_method` fixture returns an instance of this concrete class,
# and we make assertions on a MagicMock attribute *within* that instance.
# This ensures all type checks in ExposedDataWithResponder's constructor pass.
