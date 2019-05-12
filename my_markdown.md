---
title:测试markdown语法
notebook:_Index
---

### 以下内容来自[一份简明的 Markdown 笔记与教程](https://mazhuang.org/2018/09/06/markdown-intro/)
### [印象笔记对Markdown的支持](https://list.yinxiang.com/markdown/eef42447-db3f-48ee-827b-1bb34c03eb83.php)

# 测试markdown语法
## 标题2
### 标题3
#### 标题4

> 这还是一个引用, 可以连续多行。这还是一个引用, 可以连续多行。这还是一个引用, 可以连续多行。这还是一个引用, 可以连续多行。这还是一个引用, 可以连续多行。这还是一个引用, 可以连续多行。这还是一个引用, 可以连续多行。这还是一个引用, 可以连续多行。

在markdown中，空行是关键的分隔符，风格不同的section。 

在markdown中，空行是关键的分隔符，风格不同的section。
在markdown中，空行是关键的分隔符，风格不同的section。



```python
# 遍历所有行
for row in df.iterrows() :
    print row
# 遍历所有行,并包含index的值，一行作为一个tuple返回
for row in df.itertuples():
    print row

# 遍历所有列
for column in df.columns:
    print df[column]
```

- 无序列表
- 无序列表

1. 有序列表
2. 有序列表


**不知道为什么不支持可点击清单**
 - [x]  洗碗
 - [ ]  清洗油烟机
 - [ ]  拖地


一般文字**加粗**一般文字*斜体文字*一般文字



![](https://mazhuang.org/favicon.ico "favicon")


<example@gmail.com>

水平线

---

## 表格
|编号|姓名（左）|年龄（右）|性别（中）|
|-|:-|-:|:-:|
|0|张三|28|男|
|1|李四|29|男|
|2|王二麻子|30|女|

**注意上面的第二行是左中右对齐的控制**
















