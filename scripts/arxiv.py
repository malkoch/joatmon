from __future__ import print_function

import argparse
import asyncio
import datetime
import json
import re
import sys
import uuid
from urllib.parse import urlencode

import feedparser
import requests

from joatmon.assistant.job import BaseJob
from joatmon.assistant.service import BaseService
from joatmon.assistant.task import BaseTask
from joatmon.core.utility import (
    current_time,
    first_async,
    new_object_id,
    to_enumerable,
    to_list_async
)
from joatmon.decorator.message import producer
from joatmon.orm.document import (
    create_new_type,
    Document
)
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta
from joatmon.plugin.core import register
from joatmon.plugin.database.mongo import MongoDatabase
from joatmon.plugin.message.kafka import KafkaPlugin


def query(category=None, fetched=0):
    sort_by = "submittedDate"
    sort_order = "descending"

    config = json.loads(open('iva.json', 'r').read())['configs']['arxiv']

    url = config['url'] + 'query?' + urlencode(
        {
            "search_query": category, "start": 0, "max_results": 1000, "sortBy": sort_by, "sortOrder": sort_order
        }
    )
    result = feedparser.parse(url)
    if result.get('status') == 200:
        results = result['entries']
    else:
        results = []

    yield [dict(r.items()) for r in results if r.get("title", None)]


def download(link, prefer_source_tarfile=False):
    if prefer_source_tarfile:
        url = re.sub(r'/pdf/', "/src/", link)
    else:
        url = link

    response = requests.get(url)
    return response.content


class Structured(Meta):
    structured = True

    object_id = Field(uuid.UUID, nullable=False, default=new_object_id, primary=True)
    created_at = Field(datetime.datetime, nullable=False, default=current_time)
    updated_at = Field(datetime.datetime, nullable=False, default=current_time)
    deleted_at = Field(datetime.datetime, nullable=True, default=current_time)
    is_deleted = Field(bool, nullable=False, default=False)


class Author(Structured):
    __collection__ = 'author'

    name = Field(str)


class Tag(Structured):
    __collection__ = 'tag'

    name = Field(str)
    count = Field(int)


class Source(Structured):
    __collection__ = 'source'

    id = Field(str)
    title = Field(str)
    summary = Field(str)
    published = Field(datetime.datetime)
    updated = Field(datetime.datetime)


class SourceFile(Structured):
    __collection__ = 'source_file'

    source_id = Field(uuid.UUID)
    content = Field(bytes)
    type = Field(str)


class SourceLink(Structured):
    __collection__ = 'source_link'

    source_id = Field(uuid.UUID)
    link = Field(str)
    type = Field(str)


class SourceAuthor(Structured):
    __collection__ = 'source_author'

    source_id = Field(uuid.UUID)
    author_id = Field(uuid.UUID)


class SourceTag(Structured):
    __collection__ = 'source_tag'

    source_id = Field(uuid.UUID)
    tag_id = Field(uuid.UUID)


Author = create_new_type(Author, (Document,))
Tag = create_new_type(Tag, (Document,))
Source = create_new_type(Source, (Document,))
SourceFile = create_new_type(SourceFile, (Document,))
SourceLink = create_new_type(SourceLink, (Document,))
SourceAuthor = create_new_type(SourceAuthor, (Document,))
SourceTag = create_new_type(SourceTag, (Document,))


class Task(BaseTask):
    def __init__(self, api=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('--list', dest='list', action='store_true')
        parser.add_argument('--init', dest='init', action='store_true')
        parser.add_argument('--reinit', dest='reinit', action='store_true')
        parser.add_argument('--background', dest='background', action='store_true')
        parser.set_defaults(list=False)
        parser.set_defaults(init=False)
        parser.set_defaults(reinit=False)
        parser.set_defaults(background=False)

        namespace, _ = parser.parse_known_args(sys.argv)

        super(Task, self).__init__(api, namespace.background, 1, 100)

        self.action = None
        if namespace.list:
            self.action = ['list']
        elif namespace.init:
            self.action = ['init']
        elif namespace.reinit:
            self.action = ['reinit']

        # db_config = json.loads(open('iva.json', 'r').read())['configs']['arxiv']['mongo']
        self.database = MongoDatabase('mongodb://malkoch:malkoch@127.0.0.1:27017/?replicaSet=rs0', 'arxiv')

    def reinit(self):
        # self.database.drop_database()
        for doc_type in Document.__subclasses__():
            asyncio.run(self.database.delete(doc_type, {}))

        self.init()

    def init(self):
        for tag in [
            'cs.AI', 'cs.CL', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.GT', 'cs.CV', 'cs.CY', 'cs.CR', 'cs.DS',
            'cs.DB', 'cs.DL', 'cs.DM', 'cs.DC', 'cs.ET', 'cs.FL', 'cs.GL', 'cs.GR', 'cs.AR', 'cs.HC',
            'cs.IR', 'cs.IT', 'cs.LO', 'cs.LG', 'cs.MS', 'cs.MA', 'cs.MM', 'cs.NI', 'cs.NE', 'cs.NA',
            'cs.OS', 'cs.OH', 'cs.PF', 'cs.PL', 'cs.RO', 'cs.SI', 'cs.SE', 'cs.SD', 'cs.SC', 'cs.SY'
        ]:
            tag = Tag(**{'object_id': uuid.uuid4(), 'name': tag, 'count': 0})
            asyncio.run(self.database.insert(Tag, tag))

    @staticmethod
    def help(api):
        ...

    def run(self):
        try:
            if self.action is None:
                raise ValueError(f'arguments are not recognized')

            if self.action[0] == 'list':
                self.api.output(asyncio.run(to_list_async(self.database.read(Tag, {}))))
            elif self.action[0] == 'init':
                self.init()
            elif self.action[0] == 'reinit':
                self.reinit()
            else:
                raise ValueError(f'arguments are not recognized')

            if not self.event.is_set():
                self.event.set()

            super(Task, self).run()
        except:
            ...


class Job(BaseJob):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--fetch', type=str)
        parser.add_argument('--download', type=str)

        namespace, _ = parser.parse_known_args(sys.argv)

        super(Job, self).__init__(1, 100)

        self.action = None
        if namespace.fetch:
            self.action = ['fetch', namespace.fetch]
        elif namespace.download:
            self.action = ['download', namespace.download]

        # db_config = json.loads(open('iva.json', 'r').read())['configs']['arxiv']['mongo']
        self.database = MongoDatabase('mongodb://malkoch:malkoch@127.0.0.1:27017/?replicaSet=rs0', 'arxiv')

        kafka_topic = json.loads(open('iva.json', 'r').read())['configs']['arxiv']['kafka_topic']
        kafka_uri = json.loads(open('iva.json', 'r').read())['configs']['arxiv']['kafka_uri']
        register(KafkaPlugin, 'arxiv_kafka_plugin', kafka_uri)

        self.create_event = producer('arxiv_kafka_plugin', kafka_topic)(self.create_event)

    def create_event(self, source):
        ...

    def add_pdf(self, pdf, tag):
        pdf_object_id = uuid.uuid4()

        pdf_id = pdf['id']
        pdf_updated = datetime.datetime(*pdf['updated_parsed'][:6])
        pdf_published = datetime.datetime(*pdf['published_parsed'][:6])
        pdf_title = pdf['title']
        pdf_summary = pdf['summary']

        pdf_links = [{'link': link['href'], 'type': link['type']} for link in pdf['links']]
        pdf_authors = [author['name'] for author in pdf['authors']]

        pdf_result = asyncio.run(first_async(self.database.read(Source, {'id': pdf_id})))
        if pdf_result:
            pdf_object_id = pdf_result.object_id
        else:
            source = Source(**{'object_id': pdf_object_id, 'id': pdf_id, 'published': pdf_published, 'updated': pdf_updated, 'title': pdf_title, 'summary': pdf_summary})
            asyncio.run(self.database.insert(Source, source))
            self.create_event(to_enumerable(source, string=True))

            for link in pdf_links:
                asyncio.run(self.database.insert(SourceLink, {'object_id': uuid.uuid4(), 'source_id': pdf_object_id, 'link': link['link'], 'type': link['type']}))

            for author in pdf_authors:
                author_result = asyncio.run(first_async(self.database.read(Author, {'name': author})))
                if author_result:
                    author_object_id = author_result.object_id
                else:
                    author_object_id = uuid.uuid4()
                    asyncio.run(self.database.insert(Author, {'object_id': author_object_id, 'name': author}))

                asyncio.run(self.database.insert(SourceAuthor, {'object_id': uuid.uuid4(), 'source_id': pdf_object_id, 'author_id': author_object_id}))

        tag_object = asyncio.run(first_async(self.database.read(Tag, {'name': tag})))
        if tag_object:
            tag_object_id = tag_object.object_id
        else:
            tag_object_id = uuid.uuid4()
            asyncio.run(self.database.insert(Tag, {'object_id': tag_object_id, 'name': tag, 'count': 0}))

        if not asyncio.run(first_async(self.database.read(SourceTag, {'source_id': pdf_object_id, 'tag_id': tag_object_id}))):
            asyncio.run(self.database.insert(SourceTag, {'object_id': uuid.uuid4(), 'source_id': pdf_object_id, 'tag_id': tag_object_id}))
            count = len(asyncio.run(to_list_async(self.database.read(SourceTag, {'tag_id': tag_object_id}))))

            source_tag = asyncio.run(first_async(self.database.read(Tag, {'object_id': tag_object_id})))
            source_tag.count = count
            asyncio.run(self.database.update(Tag, {'object_id': source_tag.object_id}, source_tag))

    @staticmethod
    def help(api):
        ...

    def run(self):
        try:
            if self.action is None:
                raise ValueError(f'arguments are not recognized')

            if self.action[0] == 'fetch':
                # fetch category that is wanted
                # fetch all categories

                if self.action[1] != 'all':
                    tag = asyncio.run(first_async(self.database.read(Tag, {'name': self.action[1]})))
                    if tag is not None:
                        tag_count = tag.count
                    else:
                        raise ValueError(f'{self.action[1]} is not found')
                        # tag_count = 0

                    for results in query(self.action[1], fetched=tag_count):
                        for result in results:
                            self.add_pdf(result, self.action[1])

                        if self.event.is_set():
                            break
                else:
                    for tag in asyncio.run(to_list_async(self.database.read(Tag, {}))):
                        for results in query(tag.name, fetched=tag.count):
                            for result in results:
                                self.add_pdf(result, tag.name)

                            if self.event.is_set():
                                break
            elif self.action[0] == 'download':
                # fetch category that is wanted
                # fetch all categories

                if self.action[1] != 'all':
                    tag = asyncio.run(first_async(self.database.read(Tag, {'name': self.action[1]})))

                    for source_tag in asyncio.run(to_list_async(self.database.read(SourceTag, {'tag_id': tag.object_id}))):
                        source = asyncio.run(first_async(self.database.read(Source, {'object_id': source_tag.source_id})))
                        if source is None:
                            continue

                        if self.event.is_set():
                            break
                else:
                    for source in asyncio.run(to_list_async(self.database.read(Source, {}))):
                        pdf_link = asyncio.run(first_async(self.database.read(SourceLink, {'source_id': source.object_id, 'type': 'application/pdf'})))
                        if pdf_link is None:
                            continue

                        # filename = '_'.join(re.findall(r'\w+', source.title))
                        # filename = "%s.%s" % (filename, pdf_link.link.split('/')[-1])

                        if asyncio.run(first_async(self.database.read(SourceFile, {'source_id': source.object_id}))) is not None:
                            # with open(os.path.join(r'X:\Cloud\OneDrive\WORK\Source', filename + '.pdf'), 'wb') as file:
                            #     file.write(self.orm.read_one(SourceFile, source_id=source.object_id).content)

                            continue

                        try:
                            print(f'downloading {pdf_link.link}')
                            content = download(pdf_link.link)
                            asyncio.run(self.database.insert(SourceFile(**{'source_id': source.object_id, 'content': content, 'type': 'pdf'})))
                            # with open(os.path.join(r'X:\Cloud\OneDrive\WORK\Source', filename + '.pdf'), 'wb') as file:
                            #     file.write(content)
                        except Exception as ex:
                            print(str(ex))

                        if self.event.is_set():
                            break
            else:
                raise ValueError(f'arguments are not recognized')

            if not self.event.is_set():
                self.event.set()

            super(Job, self).run()
        except:
            ...


class Service(BaseService):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--fetch', type=str)
        parser.add_argument('--download', type=str)

        namespace, _ = parser.parse_known_args(sys.argv)

        super(Service, self).__init__(1, 100)

        self.action = None
        if namespace.fetch:
            self.action = ['fetch', namespace.fetch]
        elif namespace.download:
            self.action = ['download', namespace.download]

        # db_config = json.loads(open('iva.json', 'r').read())['configs']['arxiv']['mongo']
        self.database = MongoDatabase('mongodb://malkoch:malkoch@127.0.0.1:27017/?replicaSet=rs0', 'arxiv')

    def add_pdf(self, pdf, tag):
        pdf_object_id = uuid.uuid4()

        pdf_id = pdf['id']
        pdf_updated = datetime.datetime(*pdf['updated_parsed'][:6])
        pdf_published = datetime.datetime(*pdf['published_parsed'][:6])
        pdf_title = pdf['title']
        pdf_summary = pdf['summary']

        pdf_links = [{'link': link['href'], 'type': link['type']} for link in pdf['links']]
        pdf_authors = [author['name'] for author in pdf['authors']]

        pdf_result = asyncio.run(first_async(self.database.read(Source, {'id': pdf_id})))
        if pdf_result:
            pdf_object_id = pdf_result.object_id
        else:
            asyncio.run(
                self.database.insert(
                    Source,
                    {'object_id': pdf_object_id, 'id': pdf_id, 'published': pdf_published, 'updated': pdf_updated, 'title': pdf_title, 'summary': pdf_summary}
                )
            )

            for link in pdf_links:
                asyncio.run(self.database.insert(SourceLink, {'object_id': uuid.uuid4(), 'source_id': pdf_object_id, 'link': link['link'], 'type': link['type']}))

            for author in pdf_authors:
                author_result = asyncio.run(first_async(self.database.read(Author, {'name': author})))
                if author_result:
                    author_object_id = author_result.object_id
                else:
                    author_object_id = uuid.uuid4()
                    asyncio.run(self.database.insert(Author, {'object_id': author_object_id, 'name': author}))

                asyncio.run(self.database.insert(SourceAuthor, {'object_id': uuid.uuid4(), 'source_id': pdf_object_id, 'author_id': author_object_id}))

        tag_object = asyncio.run(first_async(self.database.read(Tag, {'name': tag})))
        if tag_object:
            tag_object_id = tag_object.object_id
        else:
            tag_object_id = uuid.uuid4()
            asyncio.run(self.database.insert(Tag, {'object_id': tag_object_id, 'name': tag, 'count': 0}))

        if not asyncio.run(first_async(self.database.read(SourceTag, {'source_id': pdf_object_id, 'tag_id': tag_object_id}))):
            asyncio.run(self.database.insert(SourceTag, {'object_id': uuid.uuid4(), 'source_id': pdf_object_id, 'tag_id': tag_object_id}))
            count = len(asyncio.run(to_list_async(self.database.read(SourceTag, {'tag_id': tag_object_id}))))

            source_tag = asyncio.run(first_async(self.database.read(Tag, {'object_id': tag_object_id})))
            source_tag.count = count
            asyncio.run(self.database.update(Tag, {'object_id': source_tag.object_id}, source_tag))

    @staticmethod
    def help(api):
        ...

    def run(self):
        try:
            while True:
                if self.event.is_set():
                    break

                if self.action is None:
                    raise ValueError(f'arguments are not recognized')

                if self.action[0] == 'fetch':
                    # fetch category that is wanted
                    # fetch all categories

                    if self.action[1] != 'all':
                        tag = asyncio.run(first_async(self.database.read(Tag, {'name': self.action[1]})))
                        if tag is not None:
                            tag_count = tag.count
                        else:
                            raise ValueError(f'{self.action[1]} is not found')
                            # tag_count = 0

                        for results in query(self.action[1], fetched=tag_count):
                            for result in results:
                                self.add_pdf(result, self.action[1])

                            if self.event.is_set():
                                break
                    else:
                        for tag in asyncio.run(to_list_async(self.database.read(Tag, {}))):
                            for results in query(tag.name, fetched=tag.count):
                                for result in results:
                                    self.add_pdf(result, tag.name)

                                if self.event.is_set():
                                    break
                elif self.action[0] == 'download':
                    # fetch category that is wanted
                    # fetch all categories

                    if self.action[1] != 'all':
                        tag = asyncio.run(first_async(self.database.read(Tag, {'name': self.action[1]})))

                        for source_tag in asyncio.run(to_list_async(self.database.read(SourceTag, {'tag_id': tag.object_id}))):
                            source = asyncio.run(first_async(self.database.read(Source, {'object_id': source_tag.source_id})))
                            if source is None:
                                continue

                            if self.event.is_set():
                                break
                    else:
                        for source in asyncio.run(to_list_async(self.database.read(Source, {}))):
                            pdf_link = asyncio.run(first_async(self.database.read(SourceLink, {'source_id': source.object_id, 'type': 'application/pdf'})))
                            if pdf_link is None:
                                continue

                            # filename = '_'.join(re.findall(r'\w+', source.title))
                            # filename = "%s.%s" % (filename, pdf_link.link.split('/')[-1])

                            if asyncio.run(first_async(self.database.read(SourceFile, {'source_id': source.object_id}))) is not None:
                                # with open(os.path.join(r'X:\Cloud\OneDrive\WORK\Source', filename + '.pdf'), 'wb') as file:
                                #     file.write(self.orm.read_one(SourceFile, source_id=source.object_id).content)

                                continue

                            try:
                                print(f'downloading {pdf_link.link}')
                                content = download(pdf_link.link)
                                asyncio.run(self.database.insert(SourceFile(**{'source_id': source.object_id, 'content': content, 'type': 'pdf'})))
                                # with open(os.path.join(r'X:\Cloud\OneDrive\WORK\Source', filename + '.pdf'), 'wb') as file:
                                #     file.write(content)
                            except Exception as ex:
                                print(str(ex))

                            if self.event.is_set():
                                break
                else:
                    raise ValueError(f'arguments are not recognized')

            if not self.event.is_set():
                self.event.set()

            super(Service, self).run()
        except:
            ...


if __name__ == '__main__':
    Task(None).run()
