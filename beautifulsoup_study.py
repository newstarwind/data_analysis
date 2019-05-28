# %%

'''
Beautifulsoup practice
Beautifulsoup 用于解析HTML文档
文档： https://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/
'''
import urllib2
from bs4 import BeautifulSoup

html_doc = '''
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
'''
soup = BeautifulSoup(html_doc)
print soup.prettify() # print formated html
# %%
'''
遍历直接子节点
'''
head = soup.contents[0]
print head.name # the root of html tree structure
for child in head.children: # go through direct child of head
    print(child)
# %%
'''
遍历所有后代节点
'''
for descends in head.descendants: # go through all child tree of head Tag
    print descends

# %%
'''
Beautiful Soup将复杂HTML文档转换成一个复杂的树形结构,每个节点都是Python对象, Tag 是一个重要对象
Find one Tag and its attributes
'''
print soup.title # Get a Tag title
print soup.title.name # Tag name
print soup.title.string # Tag text
print soup.title.parent.name
#%%
print soup.p # Get a Tag p
print soup.p.name
print soup.p.text
print soup.p['class'] # get attribute 
print soup.p.parent.name
#%%
print soup.a # if there are many a, return the first one
print soup.a.text
print soup.a.attrs # get all attributes of a Tag
for att in soup.a.attrs:
    print att,soup.a.attrs[att]

#%%
'''
find_all 是 核心方法，可以定位并返回找到的Tag

find_all( name , attrs , recursive , text , limit, **kwargs )
* name: Tag name, for example 'a','head','p'
* attrs: Tag attributes, format: attributes name = value, eg: id='link', class_='sister'
* recursive: whether go through all sub child for just direct child, defeat True
* text: the text value in Tag, format: text = value, for example: text = 'Elsie'
* limit: define the nunber of search result, if limit = 1, only return the first search result

'''
for link in soup.find_all('a'):
    print link.text
for row in soup.find_all('b'):
    print row.text
#%%
'''
Get all text in html
'''
print soup.get_text()
#%%
'''
Find Tag by re statement
'''
import re
for row in soup.find_all(re.compile('^t')):
    print row
#%%
'''
Find Tag 
* by attributes
* by re conditions
'''
for row in soup.find_all(href=re.compile("elsie")):
    print row

for row in soup.find_all(href = re.compile('elsie'), id = 'link1'):
    print row


#%%
'''
按照CSS类名搜索tag, 但标识CSS类名的关键字 class 在Python中是保留字,使用 class 做参数会导致语法错误.
可以通过 class_ 参数搜索有指定CSS类名的tag
'''
for row in soup.find_all('a', class_ = 'sister'):
    print row

#%%
'''
通过 text 参数可以搜搜文档中的字符串内容
可以使用正则表达式匹配内容
'''
for row in soup.find_all(text= ['Elsie','Tillie','Lacie']):
    print row

# re match
for row in soup.find_all(text = re.compile('Dormouse')):
    print row

for row in soup.find_all('a', text = 'Elsie'):
    print row    

#%%
'''
使用 limit 参数限制返回结果的数量，提升效率
使用 recursive 参数决定是否遍历所有子节点(True) 还是直接子节点(False)，默认为True
'''
for row in soup.find_all('a',limit = 2, recursive= True):
    print row

#%%
