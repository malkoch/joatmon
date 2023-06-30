from __future__ import print_function

import asyncio
import datetime
import json
import re
import time
import uuid
from urllib.parse import urlencode

import feedparser
import requests

from joatmon.assistant.service import BaseService
from joatmon.assistant.task import BaseTask
from joatmon.utility import (
    current_time,
    first_async,
    new_object_id,
    to_enumerable,
    to_list_async
)
from joatmon.decorator.message import (
    consumer,
    producer
)
from joatmon.orm.document import (
    create_new_type,
    Document
)
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta
from joatmon.plugin.core import register
from joatmon.plugin.database.mongo import MongoDatabase
from joatmon.plugin.message.kafka import KafkaPlugin


def query(category=None, fetched=0, count=10):
    sort_by = "submittedDate"
    sort_order = "descending"

    config = json.loads(open('iva.json', 'r').read())['configs']['arxiv']

    url = config['url'] + 'query?' + urlencode(
        {
            "search_query": category, "start": 0, "max_results": count, "sortBy": sort_by, "sortOrder": sort_order
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
#
#
# class Task(BaseTask):
#     def __init__(self, api, **kwargs):
#         super(Task, self).__init__(api, **kwargs)
#
#         self.database = MongoDatabase('mongodb://malkoch:malkoch@127.0.0.1:27017/?replicaSet=rs0', 'arxiv')
#
#     @staticmethod
#     def create(api):
#         return {}
#
#     def reinit(self):
#         # self.database.drop_database()
#         for doc_type in Document.__subclasses__():
#             asyncio.run(self.database.delete(doc_type, {}))
#
#         self.init()
#
#     def init(self):
#         for tag in [
#             'cs.AI', 'cs.CL', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.GT', 'cs.CV', 'cs.CY', 'cs.CR', 'cs.DS',
#             'cs.DB', 'cs.DL', 'cs.DM', 'cs.DC', 'cs.ET', 'cs.FL', 'cs.GL', 'cs.GR', 'cs.AR', 'cs.HC',
#             'cs.IR', 'cs.IT', 'cs.LO', 'cs.LG', 'cs.MS', 'cs.MA', 'cs.MM', 'cs.NI', 'cs.NE', 'cs.NA',
#             'cs.OS', 'cs.OH', 'cs.PF', 'cs.PL', 'cs.RO', 'cs.SI', 'cs.SE', 'cs.SD', 'cs.SC', 'cs.SY'
#         ]:
#             tag = Tag(**{'object_id': uuid.uuid4(), 'name': tag, 'count': 0})
#             asyncio.run(self.database.insert(Tag, tag))
#
#     def run(self):
#         self.api.output(asyncio.run(to_list_async(self.database.read(Tag, {}))))
#
#         if not self.event.is_set():
#             self.event.set()


class Task(BaseTask):
    def __init__(self, api=None, **kwargs):
        super(Task, self).__init__(api, True, **kwargs)

        self.database = MongoDatabase('mongodb://malkoch:malkoch@127.0.0.1:27017/?replicaSet=rs0', 'arxiv')

        self.create_event = producer('kafka_plugin', 'arxiv')(self.create_event)

    @staticmethod
    def help():
        return ''

    @staticmethod
    def params():
        return ['mode', 'tag']

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

    def run(self):
        mode = self.kwargs.get('mode', '')
        subject = self.kwargs.get('subject', '')
        count = int(self.kwargs.get('count', '1'))

        if mode == 'fetch':
            # fetch category that is wanted
            # fetch all categories

            tag = asyncio.run(first_async(self.database.read(Tag, {'name': subject})))
            if tag is not None:
                tag_count = tag.count
            else:
                raise ValueError(f'{subject} is not found')
                # tag_count = 0

            for results in query(subject, fetched=tag_count, count=count):
                for result in results:
                    self.add_pdf(result, subject)

                if self.event.is_set():
                    break
        if mode == 'download':
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

        if not self.event.is_set():
            self.event.set()


class Service(BaseService):
    arguments = {}

    def __init__(self, api=None, **kwargs):
        super(Service, self).__init__(api, **kwargs)

        self.database = MongoDatabase('mongodb://malkoch:malkoch@127.0.0.1:27017/?replicaSet=rs0', 'arxiv')

        register(KafkaPlugin, 'kafka_plugin', 'localhost:9092')

        self.create_event = consumer('kafka_plugin', 'arxiv')(self.create_event)
        self.send_mail = producer('kafka_plugin', 'mail')(self.send_mail)

        self.last_mail_time = datetime.datetime.now()

        self.buffer = []

    @staticmethod
    def create(api):
        return {}

    def create_event(self, source):
        self.buffer.append(source)

    def send_mail(self, mail):
        ...

    def run(self):
        while True:
            if self.event.is_set():
                break
            time.sleep(1)

            if datetime.datetime.now() - self.last_mail_time > datetime.timedelta(seconds=3600):
                self.last_mail_time = datetime.datetime.now()

                if len(self.buffer) == 0:
                    continue

                config = json.loads(open('iva.json', 'r').read())['configs']['arxiv']
                receivers = config['receivers']

                mail = {
                    'receivers': receivers,
                    'message': """    
                <html>
                    <head></head>
                    <body>
                """ + '<br><br>'.join(
                        [
                            f'<b>Title:</b> {source["title"]}<br>'
                            f'<b>Summary:</b> {source["summary"]}<br>'
                            f'<b>Link:</b> {source["id"]}<br>'
                            f'<b>Publish Date:</b> {source["published"]}' for source in self.buffer]
                    ) + """
                    </body>
                </html>
                """,
                    'subject': 'Arxiv',
                    'content_type': 'html'
                }

                self.send_mail(mail)

                self.buffer = []


if __name__ == '__main__':
    Task(None).run()
