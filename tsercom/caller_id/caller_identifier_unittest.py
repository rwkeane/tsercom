import sys
import uuid
import pytest
from unittest.mock import MagicMock

# --- Start of Protobuf Mocking ---
# Define a mock class that will act as tsercom.caller_id.proto.CallerId
# This class can be instantiated and will have an 'id' attribute.
class MockProtoCallerIdType:
    def __init__(self, id=None):
        self.id = id

    def __call__(self, *args, **kwargs): # To allow instance to be callable if it was a MagicMock before
        return self # Not ideal, but trying to cover bases if it was used as a callable mock

# Mock the 'tsercom.caller_id.proto' module and its 'CallerId' attribute.
mock_proto_module = MagicMock()
mock_proto_module.CallerId = MockProtoCallerIdType

# Ensure 'tsercom' and 'tsercom.caller_id' are in sys.modules as MagicMocks if not present
if 'tsercom' not in sys.modules:
    sys.modules['tsercom'] = MagicMock()
if 'tsercom.caller_id' not in sys.modules:
    sys.modules['tsercom.caller_id'] = MagicMock()

sys.modules['tsercom.caller_id.proto'] = mock_proto_module
# --- End of Protobuf Mocking ---

# Now import the module to be tested.
# Its 'from tsercom.caller_id.proto import CallerId' should now get our MockProtoCallerIdType.
from tsercom.caller_id.caller_identifier import CallerIdentifier


def test_random():
    identifier = CallerIdentifier.random()
    assert isinstance(identifier, CallerIdentifier)
    assert isinstance(identifier._CallerIdentifier__id, uuid.UUID)

def test_try_parse_valid_uuid_string():
    valid_uuid_str = str(uuid.uuid4())
    identifier = CallerIdentifier.try_parse(valid_uuid_str)
    assert isinstance(identifier, CallerIdentifier)
    assert str(identifier._CallerIdentifier__id) == valid_uuid_str

def test_try_parse_invalid_uuid_string():
    identifier = CallerIdentifier.try_parse("not-a-uuid")
    assert identifier is None

def test_try_parse_grpc_caller_id():
    valid_uuid_str = str(uuid.uuid4())
    # Create an instance of our mocked CallerId type
    mock_grpc_id = MockProtoCallerIdType(id=valid_uuid_str)

    identifier = CallerIdentifier.try_parse(mock_grpc_id)
    assert isinstance(identifier, CallerIdentifier), "Parsing valid mock gRPC ID failed"
    if identifier: # Pytest warning suppression
        assert str(identifier._CallerIdentifier__id) == valid_uuid_str

def test_try_parse_non_string_non_grpc():
    identifier = CallerIdentifier.try_parse(12345)
    assert identifier is None

def test_to_grpc_type():
    identifier = CallerIdentifier.random()
    grpc_id = identifier.to_grpc_type() # This should now return an instance of MockProtoCallerIdType

    assert isinstance(grpc_id, MockProtoCallerIdType)
    assert grpc_id.id == str(identifier._CallerIdentifier__id)

def test_equality():
    uuid_val = uuid.uuid4()
    id1 = CallerIdentifier(uuid_val)
    id2 = CallerIdentifier(uuid_val)
    id3 = CallerIdentifier.random()

    assert id1 == id2
    assert id1 != id3
    assert id2 != id3
    assert (id1 == str(uuid_val)) is False
    assert (id1 != None) is True

def test_hash():
    uuid_val = uuid.uuid4()
    id1 = CallerIdentifier(uuid_val)
    id2 = CallerIdentifier(uuid_val)
    id3 = CallerIdentifier.random()

    assert hash(id1) == hash(id2)
    if id1._CallerIdentifier__id != id3._CallerIdentifier__id:
        assert hash(id1) != hash(id3)

def test_string_representations():
    identifier = CallerIdentifier.random()
    uuid_str = str(identifier._CallerIdentifier__id)

    assert str(identifier) == uuid_str
    assert repr(identifier) == f"CallerIdentifier('{uuid_str}')"
    assert format(identifier) == uuid_str
    assert f"{identifier}" == uuid_str
    assert f"{identifier!s}" == uuid_str
    assert f"{identifier!r}" == f"CallerIdentifier('{uuid_str}')"

def test_try_parse_grpc_caller_id_with_non_string_id_attr():
    mock_grpc_id = MockProtoCallerIdType(id=12345) # ID is not a string
    identifier = CallerIdentifier.try_parse(mock_grpc_id)
    assert identifier is None

def test_try_parse_grpc_caller_id_with_invalid_uuid_string_id_attr():
    mock_grpc_id = MockProtoCallerIdType(id="not-a-valid-uuid")
    identifier = CallerIdentifier.try_parse(mock_grpc_id)
    assert identifier is None

def test_try_parse_object_without_id_attr():
    class NoIdAttr:
        pass
    obj = NoIdAttr()
    identifier = CallerIdentifier.try_parse(obj)
    assert identifier is None

def test_constructor_with_uuid_object():
    u = uuid.uuid4()
    identifier = CallerIdentifier(u)
    assert identifier._CallerIdentifier__id == u

def test_constructor_with_invalid_type():
    # Check specific error message if desired, but pytest.raises is good for now.
    with pytest.raises(TypeError, match="id must be a UUID instance, got <class 'str'>"):
        CallerIdentifier("not-a-uuid-object")
    with pytest.raises(TypeError, match="id must be a UUID instance, got <class 'int'>"):
        CallerIdentifier(123)

# Removed redundant/problematic tests like:
# - test_try_parse_grpc_caller_id_correct_access (covered by test_try_parse_grpc_caller_id)
# - test_to_grpc_type_mock_instance_check (covered by test_to_grpc_type with refined mock)
# - test_id_property (testing private attribute)
# The original test_try_parse_grpc_caller_id was failing due to mock issues, now hopefully fixed.
# The original test_to_grpc_type was failing due to mock issues, now hopefully fixed.
# The original test_string_representations for repr was failing, fixed in SUT.
# The original test_constructor_with_invalid_type was expecting TypeError, SUT raised AssertionError, fixed in SUT.
# The original test_to_grpc_type_mock_instance_check was checking isinstance(grpc_id, MagicMock), now uses MockProtoCallerIdType.
