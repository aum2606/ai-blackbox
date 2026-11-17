import os

from alembic import context
from evidence_store.engine import make_engine

url = context.config.get_main_option("sqlalchemy.url") or os.environ["DATABASE_URL"]

with make_engine(url).connect() as connection:
    context.configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()
