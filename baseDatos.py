# db_connection.py
from sqlalchemy import create_engine, MetaData, Table, select

class DBConnection:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        self.metadata = MetaData()

    def get_categories(self):
        """
        Carga todas las categorías de la tabla 'categories'.
        """
        category_table = Table('categories', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as connection:
            query = select(category_table.c.id, category_table.c.category_name)
            result = connection.execute(query).fetchall()
            return [(row[0], row[1]) for row in result]

    def get_prefixes_by_category(self, category_id):
        """
        Carga todos los prefijos relacionados con una categoría específica.
        """
        prefix_table = Table('prefixes', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as connection:
            query = select(prefix_table.c.prefix).where(prefix_table.c.category_id == category_id)
            result = connection.execute(query).fetchall()
            return [row[0] for row in result]

    def get_suffixes_by_category(self, category_id):
        """
        Carga todos los sufijos relacionados con una categoría específica.
        """
        suffix_table = Table('suffixes', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as connection:
            query = select(suffix_table.c.suffix).where(suffix_table.c.category_id == category_id)
            result = connection.execute(query).fetchall()
            return [row[0] for row in result]
