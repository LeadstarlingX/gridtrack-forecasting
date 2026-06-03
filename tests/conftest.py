import pytest
from testcontainers.rabbitmq import RabbitMqContainer


@pytest.fixture(scope="session")
def rabbitmq_url():
    with RabbitMqContainer("rabbitmq:management-alpine") as container:
        yield container.get_connection_url()
