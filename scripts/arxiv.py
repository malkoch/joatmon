from __future__ import print_function

import argparse
import datetime
import json
import re
import sys
import time
import uuid
from urllib.parse import urlencode

import feedparser
import requests

from joatmon.assistant.task import BaseTask
from joatmon.orm.constraint import UniqueConstraint
from joatmon.orm.document import (
    create_new_type,
    Document
)
from joatmon.orm.field import Field
from joatmon.orm.index import Index
from joatmon.orm.meta import Meta
from joatmon.plugin.database.mongo import MongoDatabase


def query(category=None, fetched=0):
    max_results = float('inf')
    sort_by = "submittedDate"
    sort_order = "ascending"

    total = fetched
    while max_results > total:
        if fetched >= 50000:
            fetched = fetched - 50000
            sort_order = 'descending'

        config = json.loads(open('iva.json', 'r').read())['configs']['arxiv']

        url = config['url'] + 'query?' + urlencode(
            {
                "search_query": category, "start": fetched, "max_results": min(max_results - fetched, 1000), "sortBy": sort_by, "sortOrder": sort_order
            }
        )
        result = feedparser.parse(url)
        if result.get('status') == 200:
            results = result['entries']
            max_results = int(result['feed']['opensearch_totalresults'])
        else:
            break

        total = total + len(results)
        fetched = fetched + len(results)
        yield [dict(r.items()) for r in results if r.get("title", None)]
        time.sleep(5)

        if total >= max_results:
            break


def download(link, prefer_source_tarfile=False):
    if prefer_source_tarfile:
        url = re.sub(r'/pdf/', "/src/", link)
    else:
        url = link

    response = requests.get(url)
    return response.content


class Author(Meta):
    __collection__ = 'arxiv.author'

    name = Field(str)

    unique_constraint_name = UniqueConstraint('name')
    index_name = Index('name')


class Tag(Meta):
    __collection__ = 'arxiv.tag'

    name = Field(str)
    count = Field(int)

    unique_constraint_name = UniqueConstraint('name')
    index_name = Index('name')


class Source(Meta):
    __collection__ = 'arxiv.source'

    id = Field(str)
    title = Field(str)
    summary = Field(str)
    published = Field(datetime.datetime)
    updated = Field(datetime.datetime)

    unique_constraint_id = UniqueConstraint('id')
    index_id = Index('id')


class SourceFile(Meta):
    __collection__ = 'arxiv.source_file'

    source_id = Field(uuid.UUID)
    content = Field(bytes)
    type = Field(str)

    unique_constraint_source_id = UniqueConstraint('source_id')
    index_source_id = Index('source_id')


class SourceLink(Meta):
    __collection__ = 'arxiv.source_link'

    source_id = Field(uuid.UUID)
    link = Field(str)
    type = Field(str)

    index_source_id = Index('source_id')


class SourceAuthor(Meta):
    __collection__ = 'arxiv.source_author'

    source_id = Field(uuid.UUID)
    author_id = Field(uuid.UUID)

    unique_constraint_source_id_author_id = UniqueConstraint('source_id,author_id')
    index_source_id = Index('source_id')
    index_author_id = Index('author_id')


class SourceTag(Meta):
    __collection__ = 'arxiv.source_tag'

    source_id = Field(uuid.UUID)
    tag_id = Field(uuid.UUID)

    unique_constraint_source_id_tag_id = UniqueConstraint('source_id,tag_id')
    index_source_id = Index('source_id')
    index_tag_id = Index('tag_id')


Author = create_new_type(Author, (Document,))
Tag = create_new_type(Tag, (Document,))
Source = create_new_type(Source, (Document,))
SourceFile = create_new_type(SourceFile, (Document,))
SourceLink = create_new_type(SourceLink, (Document,))
SourceAuthor = create_new_type(SourceAuthor, (Document,))
SourceTag = create_new_type(SourceTag, (Document,))


class Task(BaseTask):
    def __init__(self, api=None):
        super(Task, self).__init__(api, False, 1, 100)

        parser = argparse.ArgumentParser()
        parser.add_argument('--fetch', type=str)
        parser.add_argument('--download', type=str)
        parser.add_argument('--list', dest='list', action='store_true')
        parser.add_argument('--init', dest='init', action='store_true')
        parser.add_argument('--reinit', dest='reinit', action='store_true')
        parser.set_defaults(list=False)
        parser.set_defaults(init=False)
        parser.set_defaults(reinit=False)

        namespace, _ = parser.parse_known_args(sys.argv)

        self.action = None
        if namespace.fetch:
            self.action = ['fetch', namespace.fetch]
        elif namespace.download:
            self.action = ['download', namespace.download]
        elif namespace.list:
            self.action = ['list']
        elif namespace.init:
            self.action = ['init']
        elif namespace.reinit:
            self.action = ['reinit']

        db_config = json.loads(open('iva.json', 'r').read())['configs']['arxiv']['mongo']
        self.database = MongoDatabase('arxiv', **db_config)

    def reinit(self):
        self.database.drop_database()
        self.init()

    def init(self):
        ...
        # for tag in [
        #     'cs.AI', 'cs.CL', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.GT', 'cs.CV', 'cs.CY', 'cs.CR', 'cs.DS',
        #     'cs.DB', 'cs.DL', 'cs.DM', 'cs.DC', 'cs.ET', 'cs.FL', 'cs.GL', 'cs.GR', 'cs.AR', 'cs.HC',
        #     'cs.IR', 'cs.IT', 'cs.LO', 'cs.LG', 'cs.MS', 'cs.MA', 'cs.MM', 'cs.NI', 'cs.NE', 'cs.NA',
        #     'cs.OS', 'cs.OH', 'cs.PF', 'cs.PL', 'cs.RO', 'cs.SI', 'cs.SE', 'cs.SD', 'cs.SC', 'cs.SY'
        # ]:
        #     self.orm.save(Tag(object_id=uuid.uuid4(), name=tag, count=0))

    def add_pdf(self, pdf, tag):
        pdf_object_id = uuid.uuid4()

        pdf_id = pdf['id']
        pdf_updated = pdf['updated_parsed']
        pdf_published = pdf['published_parsed']
        pdf_title = pdf['title']
        pdf_summary = pdf['summary']

        pdf_links = [{'link': link['href'], 'type': link['type']} for link in pdf['links']]
        pdf_authors = [author['name'] for author in pdf['authors']]

        pdf_result = self.database.read_one(Source, id=pdf_id)
        if pdf_result:
            pdf_object_id = pdf_result.object_id
        else:
            self.database.save(Source(object_id=pdf_object_id, id=pdf_id, published=pdf_published, updated=pdf_updated, title=pdf_title, summary=pdf_summary))

            for link in pdf_links:
                self.database.save(SourceLink(object_id=uuid.uuid4(), source_id=pdf_object_id, link=link['link'], type=link['type']))

            for author in pdf_authors:
                author_result = self.database.read_one(Author, name=author)
                if author_result:
                    author_object_id = author_result.object_id
                else:
                    author_object_id = uuid.uuid4()
                    self.database.save(Author(object_id=author_object_id, name=author))

                self.database.save(SourceAuthor(object_id=uuid.uuid4(), source_id=pdf_object_id, author_id=author_object_id))

        tag_object = self.database.read_one(Tag, name=tag)
        if tag_object:
            tag_object_id = tag_object.object_id
        else:
            tag_object_id = uuid.uuid4()
            self.database.save(Tag(object_id=tag_object_id, name=tag, count=0))

        if not self.database.read_one(SourceTag, source_id=pdf_object_id, tag_id=tag_object_id):
            self.database.save(SourceTag(object_id=uuid.uuid4(), source_id=pdf_object_id, tag_id=tag_object_id))
            count = self.database.count(SourceTag, tag_id=tag_object_id)

            source_tag = self.database.read_one(Tag, object_id=tag_object_id)
            source_tag.count = count
            self.database.update(source_tag)

    @staticmethod
    def help(api):
        ...

    def run(self):
        if self.action is None:
            raise ValueError(f'arguments are not recognized')

        if self.action[0] == 'list':
            self.api.output(self.database.read_many(Tag))
        elif self.action[0] == 'init':
            self.init()
        elif self.action[0] == 'reinit':
            self.reinit()
        elif self.action[0] == 'fetch':
            # fetch category that is wanted
            # fetch all categories

            if self.action[1] != 'all':
                tag = self.database.read_one(Tag, name=self.action[1])
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
                for tag in self.database.read_many(Tag):
                    tag_count = tag.count

                    for results in query(tag.name, fetched=tag_count):
                        for result in results:
                            self.add_pdf(result, tag.name)

                        if self.event.is_set():
                            break
        elif self.action[0] == 'download':
            # fetch category that is wanted
            # fetch all categories

            if self.action[1] != 'all':
                tag = self.database.read_one(Tag, name=self.action[1])

                for source_tag in self.database.read_many(SourceTag, tag_id=tag.object_id):
                    source = self.database.read_one(Source, object_id=source_tag.source_id)
                    if source is None:
                        continue

                    if self.event.is_set():
                        break
            else:
                for source in self.database.read_many(Source):
                    pdf_link = self.database.read_one(SourceLink, source_id=source.object_id, type='application/pdf')
                    if pdf_link is None:
                        continue

                    # filename = '_'.join(re.findall(r'\w+', source.title))
                    # filename = "%s.%s" % (filename, pdf_link.link.split('/')[-1])

                    if self.database.read_one(SourceFile, source_id=source.object_id) is not None:
                        # with open(os.path.join(r'X:\Cloud\OneDrive\WORK\Source', filename + '.pdf'), 'wb') as file:
                        #     file.write(self.orm.read_one(SourceFile, source_id=source.object_id).content)

                        continue

                    try:
                        print(f'downloading {pdf_link.link}')
                        content = download(pdf_link.link)
                        self.database.save(SourceFile(source_id=source.object_id, content=content, type='pdf'))
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

        super(Task, self).run()


if __name__ == '__main__':
    Task(None).run()
