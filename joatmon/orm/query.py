import re
import typing


class Query:
    def __init__(self, query: dict):
        self.query = query

    def __contains__(self, item: object):
        return item in self.query

    def __getitem__(self, item: object):
        return self.query[item]

    @property
    def dict(self):
        ret = {}
        for key, value in self.query.items():
            if key == '$lookup':
                ret[key] = {k: v.list if hasattr(v, 'list') else v for k, v in value.items()}
            else:
                ret[key] = value
        return ret


class Pipeline:
    def __init__(self, pipeline: typing.List[Query]):
        self.queries = pipeline

    def insert_query(self, idx: int, query: Query):
        self.queries.insert(idx, query)

    def add_query(self, query: Query):
        self.queries.append(query)

    def __iter__(self):
        for query in self.queries:
            yield query

    @property
    def list(self):
        ret = []
        for query in self.queries:
            value = query.dict
            ret.append(value)
        return ret


class QueryBuilder:
    def __init__(self, collection: str):
        self.collection = collection
        self.pipeline = Pipeline([])

        self.keyword_mapper = {
            'cls': 'class',
            'classs': 'class'
        }
        self.aggregation = None

    def build(self):
        self.aggregation = self.pipeline.list
        return self

    def notnull(self, *fields):
        for field in fields:
            _fields = field.split('.')

            current_pipeline = self.pipeline
            for local_field in _fields[:-1]:
                found_field = False
                for q in current_pipeline:
                    if '$lookup' in q and q['$lookup']['as'] == local_field:
                        found_field = True
                        current_pipeline = q['$lookup']['pipeline']
                        break

                if not found_field:
                    raise ValueError(f'Could not found the field {field} - {local_field}')

            current_pipeline.add_query(Query({'$match': {'$expr': {'$and': [{'$ne': [f'${_fields[-1]}', None]}]}}}))

        return self

    def unwind(self, local, path):
        if local == self.collection:
            self.pipeline.add_query(Query({'$unwind': {'path': f'${path}'}}))
        else:
            local_fields = local.split('.')

            current_pipeline = self.pipeline
            for local_field in local_fields:
                found_field = False
                for q in current_pipeline:
                    if '$lookup' in q and q['$lookup']['as'] == local_field:
                        found_field = True
                        current_pipeline = q['$lookup']['pipeline']
                        break

                if not found_field:
                    raise ValueError(f'Could not found the local field {local} - {local_field}')

            current_pipeline.add_query(Query({'$unwind': {'path': f'${path}'}}))

    def replace_root(self, local, new_root):
        if local == self.collection:
            self.pipeline.add_query(Query({'$replaceRoot': {'newRoot': f'${new_root}'}}))
        else:
            local_fields = local.split('.')

            current_pipeline = self.pipeline
            for local_field in local_fields:
                found_field = False
                for q in current_pipeline:
                    if '$lookup' in q and q['$lookup']['as'] == local_field:
                        found_field = True
                        current_pipeline = q['$lookup']['pipeline']
                        break

                if not found_field:
                    raise ValueError(f'Could not found the local field {local} - {local_field}')

            current_pipeline.add_query(Query({'$replaceRoot': {'newRoot': f'${new_root}'}}))

    def join(self, local, foreign, name, **kwargs):  # join local and foreign on field with name
        let = {f'variable_{idx}': f'${k}' for idx, (k, v) in enumerate(kwargs.items())}
        reversed_let = {v: k for k, v in let.items()}
        reversed_kwargs = {v: k for k, v in kwargs.items()}
        query = {'$match': {'$expr': {'$and': [{'$eq': [f'$${reversed_let[f"${reversed_kwargs[v]}"]}', f'${v}']} for k, v in kwargs.items()]}}}

        if local == self.collection:
            self.pipeline.add_query(Query({'$lookup': {'from': foreign, 'let': let, 'pipeline': Pipeline([Query(query)]), 'as': name}}))
        else:
            local_fields = local.split('.')

            current_pipeline = self.pipeline
            for local_field in local_fields:
                found_field = False
                for q in current_pipeline:
                    if '$lookup' in q and q['$lookup']['as'] == local_field:
                        found_field = True
                        current_pipeline = q['$lookup']['pipeline']
                        break

                if not found_field:
                    raise ValueError(f'Could not found the local field {local} - {local_field}')

            current_pipeline.add_query(Query({'$lookup': {'from': foreign, 'let': let, 'pipeline': Pipeline([Query(query)]), 'as': name}}))
        return self

    def match(self, local, **kwargs):  # in, not in, equals, not equals
        if local == self.collection:
            self.pipeline.insert_query(0, Query({'$match': {'$expr': {'$and': [{'$eq': [f'${k}', v]} for k, v in kwargs.items()]}}}))
        else:
            local_fields = local.split('.')

            current_pipeline = self.pipeline
            for local_field in local_fields:
                found_field = False
                for q in current_pipeline:
                    if '$lookup' in q and q['$lookup']['as'] == local_field:
                        found_field = True
                        current_pipeline = q['$lookup']['pipeline']
                        break

                if not found_field:
                    raise ValueError(f'Could not found the local field {local} - {local_field}')

            current_pipeline.insert_query(0, Query({'$match': {'$expr': {'$and': [{'$eq': [f'${k}', v]} for k, v in kwargs.items()]}}}))

        return self

    def project(self, local, **kwargs):
        projection = {}
        for key, value in kwargs.items():
            if not isinstance(key, str):
                raise ValueError(f'{type(value)} typed keys are not supported in projection')
            _key = self.keyword_mapper.get(key, key)

            if isinstance(value, (int, dict)):
                projection[_key] = value
            elif isinstance(value, str):
                match = re.search(r'map\((.*?)\)', value)
                if match is not None:
                    maps = list(map(lambda x: x.strip(), match.group(1).split(',')))
                    projection[_key] = {'$map': {'input': f'${key}', 'as': 'dummy', 'in': {v: f'$$dummy.{v}' for v in maps}}}
                elif '@' in value:
                    collection, at = value.split('@')
                    if collection == '':
                        collection = _key
                    if at == '':
                        at = '0'
                    projection[_key] = {'$ifNull': [{'$arrayElemAt': [f'${collection}', int(at)]}, None]}
                else:
                    projection[_key] = f'${value}'
            else:
                raise ValueError(f'{type(value)} typed values are not supported in projection')

        if local == self.collection:
            self.pipeline.add_query(Query({'$project': projection}))
        else:
            local_fields = local.split('.')

            current_pipeline = self.pipeline
            for local_field in local_fields:
                found_field = False
                for q in current_pipeline:
                    if '$lookup' in q and q['$lookup']['as'] == local_field:
                        found_field = True
                        current_pipeline = q['$lookup']['pipeline']
                        break

                if not found_field:
                    raise ValueError(f'Could not found the local field {local} - {local_field}')

            current_pipeline.add_query(Query({'$project': projection}))
        return self
