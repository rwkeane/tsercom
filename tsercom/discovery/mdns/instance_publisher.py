import datetime
from uuid import getnode as get_mac

from tsercom.discovery.mdns.record_publisher import RecordPublisher


class InstancePublisher:
    """
    Publishes an instance of the specified type.
    """
    def __init__(self,
                 port : int,
                 service_type : str,
                 readable_name : str | None = None,
                 instance_name : str | None = None):
        assert isinstance(port, int), type(port)
        assert isinstance(service_type, str), type(service_type)
        assert readable_name is None or isinstance(readable_name, str), \
                type(readable_name)
        assert instance_name is None or isinstance(instance_name, str), \
                type(instance_name)

        self.__name = readable_name
        if instance_name is None:
            mac_address = get_mac()
            instance_name = f"{port}{mac_address}"
            if len(instance_name) > 15:
                instance_name = instance_name[:15]
        
        txt_record = self._make_txt_record()
        assert not txt_record is None
        self.__record_publisher = \
            RecordPublisher(instance_name,
                            service_type,
                            port,
                            txt_record)
        
    def _make_txt_record(self) -> dict[str, str | None]:
        properties : dict[str, str | None] = \
        {
            "published_on".encode('utf-8'): self.__get_current_date_time_bytes()
        }

        if not self.__name is None:
            properties["name".encode('utf-8')] = self.__name.encode('utf-8')

        return properties

    def publish(self):
        self.__record_publisher.publish()

    def __get_current_date_time_bytes(self):
        now = datetime.datetime.now()
        as_str = now.strftime('%F %T.%f')
        return as_str.encode('utf-8')
