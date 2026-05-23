"""Shared fixtures: a throwaway Postgres container (superuser) per test session."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session")
def postgres_uri() -> Iterator[str]:
    with PostgresContainer("postgres:18-alpine") as postgres:
        yield postgres.get_connection_url()
