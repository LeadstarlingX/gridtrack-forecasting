import pytest
from testcontainers.rabbitmq import RabbitMqContainer


@pytest.fixture(scope="session")
def rabbitmq_url():
    """Start a RabbitMQ container once per session and yield its AMQP URL.

    testcontainers 4.x removed get_connection_url(); we build the URL from
    get_container_host_ip() + get_exposed_port() instead.
    """
    with RabbitMqContainer("rabbitmq:management-alpine") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5672)
        yield f"amqp://guest:guest@{host}:{port}"
