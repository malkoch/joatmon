import codecs
import functools
import os
import time

import pycurl


class Download:
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, url):
        self.url = url
        self.filename = url.split('/')[-1]

        c = pycurl.Curl()
        c.setopt(pycurl.URL, self.url)
        c.setopt(pycurl.NOBODY, 1)
        c.perform()

        self.size = c.getinfo(pycurl.CONTENT_LENGTH_DOWNLOAD)
        c.close()
        del c

        self.chunks = []

        self.m = pycurl.CurlMulti()

        self.bom_check = {}

    def get_range(self, chunk, max_chunks):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if chunk == max_chunks - 1:
            return f'{int((self.size // max_chunks) * chunk)}-{int(self.size - 1)}'

        return f'{int((self.size // max_chunks) * chunk)}-{int((self.size // max_chunks) * (chunk + 1) - 1)}'

    def write_body(self, buf, chunk, f):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if not self.bom_check.get(chunk, False):
            if buf[:3] == codecs.BOM_UTF8:
                buf = buf[3:]
            self.bom_check[chunk] = True

        f.write(buf)

    def write_header(self, buf, chunk):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        print(f'{chunk}: {buf}')

    def progress(self, *args, chunk):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...
        # print(chunk, args)

    def download(self, chunks):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        files = []

        for chunk in range(chunks):
            files.append(open(f'{self.filename}.{chunk}', 'wb'))

            c = pycurl.Curl()
            c.setopt(pycurl.URL, self.url)
            c.setopt(pycurl.POST, 0)
            c.setopt(pycurl.IPRESOLVE, pycurl.IPRESOLVE_V4)
            c.setopt(pycurl.RANGE, self.get_range(chunk, chunks))
            c.setopt(pycurl.WRITEFUNCTION, functools.partial(self.write_body, chunk=chunk, f=files[chunk]))
            c.setopt(pycurl.HEADERFUNCTION, functools.partial(self.write_header, chunk=chunk))
            c.setopt(pycurl.PROGRESSFUNCTION, functools.partial(self.progress, chunk=chunk))
            c.setopt(pycurl.NOPROGRESS, 0)
            c.setopt(pycurl.POST, 0)
            # c.setopt(pycurl.HTTPGET, 1)
            c.setopt(pycurl.FOLLOWLOCATION, 1)
            c.setopt(pycurl.MAXREDIRS, 10)
            c.setopt(pycurl.CONNECTTIMEOUT, 30)
            c.setopt(pycurl.NOSIGNAL, 1)
            c.setopt(pycurl.SSL_VERIFYPEER, 0)
            c.setopt(pycurl.LOW_SPEED_TIME, 60)
            c.setopt(pycurl.LOW_SPEED_LIMIT, 5)

            c.setopt(
                pycurl.USERAGENT,
                b'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
            )
            if pycurl.version_info()[7]:
                c.setopt(pycurl.ENCODING, b'gzip, deflate')
            c.setopt(
                pycurl.HTTPHEADER,
                [
                    b'Accept: */*',
                    b'Accept-Language: en-US,en',
                    b'Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                    b'Connection: keep-alive',
                    b'Keep-Alive: 300',
                    b'Expect:',
                ],
            )
            self.chunks.append(c)

            self.m.add_handle(c)

        dones = []
        fails = []

        while True:
            while True:
                ret, num_handles = self.m.perform()
                if ret != pycurl.E_CALL_MULTI_PERFORM:
                    break

            num_q, ok_list, err_list = self.m.info_read()
            for ok in ok_list:
                dones.append(ok)
            for fail in err_list:
                fails.append(fail)
            if len(dones) + len(fails) == chunks:
                break

            self.m.select(1)
            time.sleep(0.001)

        for chunk in self.chunks:
            chunk.close()

        for file in files:
            file.close()

        f = open(self.filename, 'wb')
        for chunk in range(chunks):
            f.write(open(f'{self.filename}.{chunk}', 'rb').read())
            os.remove(f'{self.filename}.{chunk}')


def download(url, chunks):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    d = Download(url)
    d.download(chunks)
    return d
