import sys
import uuid
import pytest


# --- Start of Protobuf Mocking ---
# Define a mock class that will act as tsercom.caller_id.proto.CallerId
# This class can be instantiated and will have an 'id' attribute.
class MockProtoCallerIdType:
    def __init__(self, id=None):
        self.id = id

    def __call__(self, *args, **kwargs):
        return self


# Import the module to be tested AFTER the mock is set up by the fixture.
# This will be handled by importing CallerIdentifier within the tests or fixtures that need it,
# or by re-importing it within the fixture. For module-scoped application,
# we can reload the SUT module within the fixture.


@pytest.fixture(scope="function")
def PatchedCallerIdentifier(mocker):
    """
    Mocks the 'CallerId' protobuf class dependency within the
    tsercom.caller_id.caller_identifier module for the duration of a test.
    Returns the original CallerIdentifier class, which will now use the mocked
    protobuf class.
    """
    import tsercom.caller_id.caller_identifier  # SUT module

    # Patch the 'CallerId' name (which is an imported protobuf class)
    # directly within the tsercom.caller_id.caller_identifier module's namespace.
    mocker.patch(
        "tsercom.caller_id.caller_identifier.CallerId",  # The alias used in caller_identifier.py
        new=MockProtoCallerIdType,
    )

    # No importlib.reload is needed.
    # The CallerIdentifier class itself is not reloaded or replaced.
    # Its behavior changes because its internal reference to the protobuf class 'CallerId'
    # now points to MockProtoCallerIdType for the scope of the patch.
    return tsercom.caller_id.caller_identifier.CallerIdentifier


# --- End of Protobuf Mocking ---

# Tests will now take PatchedCallerIdentifier as an argument


def test_random(PatchedCallerIdentifier):
    identifier = PatchedCallerIdentifier.random()
    assert isinstance(identifier, PatchedCallerIdentifier)
    assert isinstance(identifier._CallerIdentifier__id, uuid.UUID)


def test_try_parse_valid_uuid_string(PatchedCallerIdentifier):
    valid_uuid_str = str(uuid.uuid4())
    identifier = PatchedCallerIdentifier.try_parse(valid_uuid_str)
    assert isinstance(identifier, PatchedCallerIdentifier)
    assert str(identifier._CallerIdentifier__id) == valid_uuid_str


def test_try_parse_invalid_uuid_string(PatchedCallerIdentifier):
    identifier = PatchedCallerIdentifier.try_parse("not-a-uuid")
    assert identifier is None


def test_try_parse_grpc_caller_id(PatchedCallerIdentifier):
    valid_uuid_str = str(uuid.uuid4())
    mock_grpc_id = MockProtoCallerIdType(
        id=valid_uuid_str
    )  # Uses the globally defined MockProtoCallerIdType

    identifier = PatchedCallerIdentifier.try_parse(mock_grpc_id)
    assert isinstance(
        identifier, PatchedCallerIdentifier
    ), "Parsing valid mock gRPC ID failed"
    if identifier:
        assert str(identifier._CallerIdentifier__id) == valid_uuid_str


def test_try_parse_non_string_non_grpc(PatchedCallerIdentifier):
    identifier = PatchedCallerIdentifier.try_parse(12345)
    assert identifier is None


def test_to_grpc_type(PatchedCallerIdentifier):
    identifier = PatchedCallerIdentifier.random()
    grpc_id = identifier.to_grpc_type()

    assert isinstance(
        grpc_id, MockProtoCallerIdType
    )  # Check against our mock type
    assert grpc_id.id == str(identifier._CallerIdentifier__id)


def test_equality(PatchedCallerIdentifier):
    uuid_val = uuid.uuid4()
    id1 = PatchedCallerIdentifier(uuid_val)
    id2 = PatchedCallerIdentifier(uuid_val)
    id3 = PatchedCallerIdentifier.random()

    assert id1 == id2
    assert id1 != id3
    assert id2 != id3
    assert (
        id1 == str(uuid_val)
    ) is False  # Comparison with non-CallerIdentifier type
    assert (id1 != None) is True  # Comparison with None


def test_hash(PatchedCallerIdentifier):
    uuid_val = uuid.uuid4()
    id1 = PatchedCallerIdentifier(uuid_val)
    id2 = PatchedCallerIdentifier(uuid_val)
    id3 = PatchedCallerIdentifier.random()

    assert hash(id1) == hash(id2)
    if (
        id1._CallerIdentifier__id != id3._CallerIdentifier__id
    ):  # Ensure they are actually different for the hash test
        assert hash(id1) != hash(id3)


def test_string_representations(PatchedCallerIdentifier):
    identifier = PatchedCallerIdentifier.random()
    uuid_str = str(identifier._CallerIdentifier__id)

    assert str(identifier) == uuid_str
    assert (
        repr(identifier) == f"CallerIdentifier('{uuid_str}')"
    )  # Note: Class name might be PatchedCallerIdentifier due to how it's passed
    assert format(identifier) == uuid_str
    assert f"{identifier}" == uuid_str
    assert f"{identifier!s}" == uuid_str
    # For repr, if PatchedCallerIdentifier is just an alias to the true reloaded class,
    # the class's __repr__ should still say 'CallerIdentifier'.
    # If the test fails here, it means `repr` is seeing the fixture name. We'll adjust if needed.
    assert f"{identifier!r}" == f"CallerIdentifier('{uuid_str}')"


def test_try_parse_grpc_caller_id_with_non_string_id_attr(
    PatchedCallerIdentifier,
):
    mock_grpc_id = MockProtoCallerIdType(id=12345)
    identifier = PatchedCallerIdentifier.try_parse(mock_grpc_id)
    assert identifier is None


def test_try_parse_grpc_caller_id_with_invalid_uuid_string_id_attr(
    PatchedCallerIdentifier,
):
    mock_grpc_id = MockProtoCallerIdType(id="not-a-valid-uuid")
    identifier = PatchedCallerIdentifier.try_parse(mock_grpc_id)
    assert identifier is None


def test_try_parse_object_without_id_attr(PatchedCallerIdentifier):
    class NoIdAttr:
        pass

    obj = NoIdAttr()
    identifier = PatchedCallerIdentifier.try_parse(obj)
    assert identifier is None


def test_constructor_with_uuid_object(PatchedCallerIdentifier):
    u = uuid.uuid4()
    identifier = PatchedCallerIdentifier(u)
    assert identifier._CallerIdentifier__id == u


def test_constructor_with_invalid_type(PatchedCallerIdentifier):
    with pytest.raises(
        TypeError, match="id_value must be a UUID instance, got <class 'str'>"
    ):
        PatchedCallerIdentifier("not-a-uuid-object")
    with pytest.raises(
        TypeError, match="id_value must be a UUID instance, got <class 'int'>"
    ):
        PatchedCallerIdentifier(123)
