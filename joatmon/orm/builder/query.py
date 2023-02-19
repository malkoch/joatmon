class Query:
    def __init__(self):
        ...

    @classmethod
    def NEW(cls):
        query = Query()
        return query

    def SELECT(self):
        ...

    def FROM(self):
        ...

    def JOIN(self):
        ...

    def WHERE(self):
        ...
