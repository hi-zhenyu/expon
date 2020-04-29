
import codecs
import pkgutil

import markdown

from .file import *
from . import gconfig


SPLIT = '\n\n'

def add_text(source, text):
    return source.append(text+SPLIT)

def add_title(source, title, level=1):
    assert type(level) == int
    assert level in [1, 2, 3, 4, 5, 6]
    title = '#'*level+' '+title
    source.append(title+SPLIT)
    return source

def add_code(source, code, lang='json'):
    code = '```'+lang+'\n'+code+'\n'+'```'
    source.append(code+SPLIT)
    return source

def add_quote(source, quote):
    quote ='> '+quote
    source.append(quote+SPLIT)
    return source

def add_line(source):
    source.append('------'+SPLIT)
    return source

def add_image(source, title, image_path):
    if check_file(image_path):
        image_str = '!['+title+']('+image_path+')'
        source.append(image_str+SPLIT)
    else:
        raise Exception('Image dose not exit!')
    return source

def save2md(source, file_path):
    if '.md' not in file_path:
        file_path += '.md'

    if check_file(file_path):
        raise Exception(file_path+'already exits!')

    md_str = ''.join(source)
    # with open(file_path, 'w') as f:
    #     f.write(source)

    with codecs.open(file_path, "w", encoding="utf-8") as f:
        f.write(md_str)

def md2html(mdstr):
    exts = ['markdown.extensions.extra', 'markdown.extensions.codehilite','markdown.extensions.tables','markdown.extensions.toc']

    html = '''
    <html lang="zh-cn">
    <head>
    <meta content="text/html; charset=utf-8" http-equiv="content-type" />
    <link href="%s" rel="stylesheet">
    <link href="%s" rel="stylesheet">
    </head>
    <body>
    %s
    </body>
    </html>
    '''

    ret = markdown.markdown(mdstr,extensions=exts)
    return html % (gconfig.DEFAULT_CSS, gconfig.GITHUB_CSS, ret)

def check_css():
    if not check_file(gconfig.DEFAULT_CSS):
        raise Exception(gconfig.DEFAULT_CSS + ' does not exit!')
        # data = pkgutil.get_data('expon', 'asset/default.css').decode('utf-8')
        # with open(gconfig.DEFAULT_CSS, 'w') as f:
        #     f.write(data)
        
    if not check_file(gconfig.GITHUB_CSS):
        raise Exception(gconfig.GITHUB_CSS + ' does not exit!')
        # data = pkgutil.get_data('expon', 'asset/github.css').decode('utf-8')
        # with open(gconfig.GITHUB_CSS, 'w') as f:
        #     f.write(data)

def save2html(source, file_path):
    check_css()
    if '.html' not in file_path:
        file_path += '.html'
    if check_file(file_path):
        raise Exception(file_path+'already exits!')

    md_str = ''.join(source)
    html = md2html(md_str)
    with codecs.open(file_path, "w", encoding="utf-8", errors="xmlcharrefreplace") as f:
        f.write(html)


