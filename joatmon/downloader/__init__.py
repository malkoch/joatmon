from joatmon.downloader.http.download import Download


def download(url, file, chunks, resume):
    d = Download(url, file, options={'interface': None, 'proxies': None, 'ipv6': None}, progress_notify=False)
    d.download(chunks, resume)
    return d
