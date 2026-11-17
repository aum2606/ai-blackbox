from alembic import context
from evidence_store.engine import database_url, make_engine

url = context.config.get_main_option("sqlalchemy.url") or database_url()

with make_engine(url).connect() as connection:
    context.configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()
