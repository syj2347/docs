# Markdown Test
## 二级标题

*倾斜字体*
**加粗字体**
***斜体加粗***
~~删除线~~

>引用

给定一个确定顺序的、由26个小写字母组成的键盘，每组给出一个单词，求出手敲完该单词所运动的距离。

---

[百度](www.baidu.com)
- 列表 
  - 列表

表头|表头|表头
:-|-|-:
内容1|内容2|内容3
靠左|内容5|靠右

`//单行代码`

```cpp
#include<bits/stdc++.h>
using namespace std;
```

#### 二分查找
```cpp
long long l = 0, r = h;
	long long  Min = 0;
	while (l <= r)
	{
		long long  mid = (l + r) / 2;
		if (check(mid))
		{
			r = mid - 1;
			Min = mid;
		}
		else
		{
			l = mid + 1;
		}
	}
	cout << Min << endl;
```

$$
\begin{cases}
x_1+x_2+x_3=7\\
2x_1-5x_2+x_3\ge10\\
x_1+3x_2+x_3\le12\\
x_1,x_2,x_3\ge0\\
\end{cases}
$$

# test
#### 复选框 checkbox
- [x] test1
- [ ] test2

