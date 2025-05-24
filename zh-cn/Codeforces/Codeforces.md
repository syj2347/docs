## Codeforces  Round #753（Div.3）A~D 题解 

### A. Linear Keyboard

#### 题意

给定一个确定顺序的、由26个小写字母组成的键盘，每组给出一个单词，求出手敲完该单词所运动的距离。

#### 思路

首先创建两个字符串a和s分别储存键盘和单词。

如果只是将键盘存在字符串中，那在过程中想寻找特定字母的位置，就需要遍历字符串寻找，数据量大的情况下有可能会超时。

所以可以创建一个整型数组k，用其下标为0到25存储a到e的对应位置，随后模拟手移动过程中查询`k[s[i]-'a']` 的值进行减法运算即可。

#### 总结

做题时需要有对储存的数据进行预处理操作的意识。

也可以用map做，自己应该尽快去学习c++相关的知识，而不是继续在c++程序中用c做题。

#### AC代码

|Problem|Lang|Verdict|Time|Memory|
|:--:|:--:|:--:|:--:|:--:|
|[A - Linear Keyboard](https://codeforces.com/contest/1607/problem/A)|GNU C++17|**Accepted**|0 ms|0 KB|

```cpp
#include<bits/stdc++.h>
using namespace std;
int t;
char a[30];
char s[55];
int k[30];
void solve()
{
    scanf("%s", a);
    scanf("%s", s);
    int sum = 0;
    memset(k, 0, sizeof(k));
    for (int i = 0; i < strlen(a); i++) {
        k[a[i] - 'a']=i;
    }
    for (int i = 1; i < strlen(s); i++) {
        sum+=abs(k[s[i]-'a'] - k[s[i - 1]-'a']);
    }
    printf("%d\n", sum);
}

int main()
{
    scanf("%d", &t);
    while (t--) {
        solve();
    }
    return 0;
}
```

---

### B. Odd Grasshopper

#### 题意

给定初始位置 $x_0 $ 和运动次数 $n$，运动的规则是如果身处的坐标为偶数，运动方向为向左，不然运动方向为向右；而运动的距离，则为其对应的运动次数，即第一次运动一个单位……第 $n$ 次运动 $n$ 个单位。求出最终位置。

共 $t$ 次测试，每次给出一组 $x_0 $ 和  $n$ ，数据范围如下：
$$
t(1≤t≤10^4)\cdots\cdots x_0(−10^{14}≤x_0≤10^{14})\cdots\cdots n(0≤n≤10^{14})
$$

#### 思路

可以发现，$x_0$ 和 $n$ 的数据范围都很大，超过了int的范围，需要注意使用long long定义变量。

另外，如此大的数据也可以提醒我们，题目中的操作应该是有周期性规律的。

于是可以发现，$T=4$ ，即每当四次操作后，会回到原处。

所以将 $n$ 取余4后，分类讨论即可。

#### 总结

当见到操作次数或数据极大时，就应该马上意识到极有可能存在周期性或有一定规律。

应当注意变量的范围，思考运算过程中会不会超int。

#### AC代码

|Problem|Lang|Verdict|Time|Memory|
|:--:|:--:|:--:|:--:|:--:|
|[B - Odd Grasshopper](https://codeforces.com/contest/1607/problem/B)|GNU C++17|**Accepted**|15 ms|0 KB|

```cpp
#include<bits/stdc++.h>
using namespace std;
int t;
long long n;
long long x;

void solve()
{
    scanf("%lld%lld", &x,&n);
    long long left = n % 4;
    if (x % 2) {
        if (left == 3)x -= n+1;
        if (left == 2)x -= 1;
        if (left == 1)x += n;
    }
    else {
        if (left == 3)x += n+1;
        if (left == 2)x += 1;
        if (left == 1)x -= n;
    }
    printf("%lld\n", x);
}

int main()
{
    scanf("%d", &t);
    while (t--) {
        solve();
    }
    return 0;
}
```

---

### C. Minimum Extraction

#### 题意

给出一个数组，可对其进行如下操作;

选择数组中最小的元素，使其余元素分别减这个元素，随后删除这个元素（若有不止一个相同的最小元素，仅删除一个）。问经过任意数量的操作后，每次操作后数组中最小的元素出现过的最大值是多少。

$t$ 组测试，数组中有 $n$ 个元素，数组元素用 $a_i$ 表示，数据范围如下：
$$
t(1≤t≤10^4)\cdots\cdots n(1≤n≤2⋅10^5)\cdots\cdots a_i (−10^9≤a_i≤10^9)
$$

#### 思路

可以将问题转化为求：将数组从小到大排序后，相邻元素差值的最大值为多少。

#### 总结

做题时最难的是看到问题的本质，需要提高转化问题的能力。

#### AC代码

|Problem|Lang|Verdict|Time|Memory|
|:--:|:--:|:--:|:--:|:--:|
|[C - Minimum Extraction](https://codeforces.com/contest/1607/problem/C)|GNU C++17|**Accepted**|78 ms|800 KB|

```cpp
#include<bits/stdc++.h>
using namespace std;
int t;
int n;
int a[200010];

void solve()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &a[i]);
    }
    sort(a, a + n);
    int Max = a[0];
    for (int i = 0; i < n-1; i++) {
        Max = max(a[i + 1] - a[i],Max);
    }
    printf("%d\n", Max);
}

int main()
{
    scanf("%d", &t);
    while (t--) {
        solve();
    }
    return 0;
}
```

---

### D. Blue-Red Permutation

#### 题意

给出一个数组，包含 $n$ 个元素，每个元素有两个属性：值和颜色，颜色分为蓝色或红色。

可有如下操作：

（1）：选择任意数量的蓝色元素，使它们的值均减一。

（2）：选择任意数量的红色元素，使它们的值均加一。

问能不能使数组最终变成 1 到 $n$ 的一个排列？

数据范围如下：

测试组数 $t$ $(1≤t≤10^4)$ ，元素个数 $n(1≤n≤2⋅10^5)$ ，元素 $a_i(−10^9≤a_i≤10^9)$

#### 思路

也就是说，蓝色元素可以变成比它小的任意值，红色元素可以变成比它大的任意值。

那么可以很容易地发现，蓝色元素的最大值需要大于等于蓝色元素的个数，同理红色元素的最小值也需要小于等于$n$ 减去红色元素的个数。

可以进一步发现，每个元素都需要符合类似的条件，不然出现类似于蓝色1、蓝色1，显然是不行的，第二个蓝色1需要大于等于2。

那么这里的个数是什么？就是将蓝色的元素从小到大排序后，该元素所处的位置，第 $n$ 个元素要大于等于 $n$ ；同理，红色元素也是类似的条件，相反即可。

所以将红蓝元素分别保存在两个数组里，排序后再遍历判断即可，全部符合输出yes，有一个不符合就可以结束遍历输出no了。

#### 总结

注意对数据的预处理操作，有时是解决问题的关键。

#### AC代码

|Problem|Lang|Verdict|Time|Memory|
|:--:|:--:|:--:|:--:|:--:|
|[D - Blue-Red Permutation](https://codeforces.com/contest/1607/problem/D)|GNU C++17|**Accepted**|78 ms|2600 KB|

```cpp
#include<bits/stdc++.h>
using namespace std;
int t;
int n;
int a[200010];
int a1[200010];
int a2[200010];
char s[200010];

void solve()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i++)
        scanf("%d", &a[i]);
    scanf("%s", s);//储存类似于“BRBR”的字符串
    int k1 = 0, k2 = 0;
    for (int i = 0; i < n; i++) {
        if (s[i] == 'B') {
            a1[k1] = a[i];
            k1++;
        }
        else {
            a2[k2] = a[i];
            k2++;
        }
    }
    sort(a1, a1 + k1);//默认为从小到大排序
    sort(a2, a2 + k2,greater<int>());//greater才是从大到小排序
    
    //测试程序
    /*for (int i = 0; i < k1; i++)
        printf("%d ", a1[i]);
    printf("\n");
    for (int i = 0; i < k2; i++)
        printf("%d ", a2[i]);*/
    
    int flag = 1;
    for (int i = 0; i < k1; i++) {
        if (a1[i] < i+1) {
            flag = 0; break;
        }
    }
    for (int i = 0; i < k2; i++) {
        if (flag == 0)break;
        if (a2[i] > n-i) {
            flag = 0; break;
        }
    }
    if (flag)printf("YES\n");
    else printf("NO\n");
}

int main()
{
    scanf("%d", &t);
    while (t--) {
        solve();
    }
    return 0;
}
```

---

##### by syj

tip：希望能有一次a了E

## Codeforces Round #768（Div.2）A~C 题解

### A. Min Max Swap

#### 题意

共 $t$ $(1≤t≤100)$ 组测试数据，每组中给出一个数组的元素个数 $n$ $(1≤n≤100) $ ，接下来两行分别是 $n$ 个元素，分别为两个数组。可以将两个数组对应位置的元素交换，使得两个数组中最大值的乘积最小，求出最小乘积。

#### 思路

> 注意读题不要读错了，不然会浪费很多时间 。

找出所有元素中最大的元素，它是一定会被乘到的，我们只需让另外一个最大值最小即可，所以我们应该使大的数尽量往最大元素所在组放，随后找出另一组的最大值即可。

也就是分别求出对应位置比较后的较大值的最大值和较小值的最大值。

#### 总结

一开始心急没仔细看，以为是可以将两个数组中的任意元素互换，浪费了不少时间。以后要好好审题。

#### AC代码

|Problem|Lang|Verdict|Time|Memory|
|:--:|:--:|:--:|:--:|:--:|
|[A - Min Max Swap](https://codeforces.com/contest/1631/problem/A)|GNU C++17|**Accepted**|15 ms|0 KB|

```cpp
#include<bits/stdc++.h>
using namespace std;
int n;
int a[110];
int b[110];
void solve()
{
    cin >> n;
    for (int i = 1; i <= n; i++)
        cin >> a[i];
    for (int i = 1; i <= n; i++)
        cin >> b[i];
    int Max1 = 0;
    int Max2 = 0;
    for (int i = 1; i <= n; i++) {
        Max1 = max(Max1, min(a[i], b[i]));
        Max2 = max(Max2, max(a[i], b[i]));
    }
    cout << Max1 * Max2 << endl;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    int tt;
    cin >> tt;
    while (tt--) {
        solve();
    }
}
```

---

### B. Fun with Even Subarrays

#### 题意

$t$ $(1≤t≤2⋅10^4)$组元素, $n$ $(1≤n≤2⋅10^5)$个元素，可进行的操作如下：

选定一个区间 $[l,r]$ ，将 $[r+1,r+r-l]$ 区间内的元素对应复制给区间 $[l,r]$ ,显然这两个区间都需在总区间内。

求出使数组中所有元素相等的最小操作数。

#### 思路

因为它是一个将右边的数复制到左边的操作，所以应该从右边开始操作。如果遇到不同的数，就进行一次操作覆盖，遇到相同的就可以继续推进，操作新覆盖的长度是等于它到起点的长度。

#### 总结

> 遇到runtime error的报错应该去检查是否存在数组越界、爆栈。

本题我当时写的是将跳转后的点的值改变成右端点，再来进行下一次比较的。那么便出现了一个问题，最后一次跳转是可能到负下标的，那么对数组负下标元素的赋值在我本地是没问题的，但在线上测试时就runtime error了。

#### AC代码

|Problem|Lang|Verdict|Time|Memory|
|:--:|:--:|:--:|:--:|:--:|
|[B - Fun with Even Subarrays](https://codeforces.com/contest/1631/problem/B)|GNU C++17|**Accepted**|108 ms|800 KB|

```cpp
#include<bits/stdc++.h>
using namespace std;
int n;
int a[200010];
void solve()
{
    cin >> n;
    for (int i = 1; i <= n; i++)
        cin >> a[i];
    int k = n;
    int cnt = 0;
    int j = 1;
    while (k>1) {
        if (a[k] == a[k - 1])k--;
        else {
            int p = k;
            k -= j;
            if(k>0)a[k] = a[p];
            //注意下标不能为负，不然会runtime error
            cnt++;
        }
        j = n - k + 1;
    }
    cout << cnt << endl;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    int tt;
    cin >> tt;
    while (tt--) {
        solve();
    }
}
```

---

### C. And Matching

#### 题意

$t$ $(1≤t≤400)$组测试，每组给出一个 $n$ 和 $k$ ，$4≤n≤2^{16}, 0≤k≤n−1$ ,$n=2^j,j∈Z$

对于 $0$ 到 $n-1$ 的这些数，将其两两分组，使得各组的两个数 `a&b` 的总和等于 $k$ ，输出任意一种分组方式，若不行，输出-1。  

#### 思路

数据很大，不能暴力枚举，那显然就是一道构造题。

分类讨论，先看一种比较特殊的情况，即 $n=0$ 时，需要使得所有 `a&b` 均为 0。

 设 $x$ 的组员应为 $c(x)$ ，那么怎么求 $c(x)$ 呢？

由于同一为一，不然则为0，所以不同的时候一定是0那么就可以将  $x$ 按位取反，求得 $c(x)$ 

又可以发现由于 $n=2^j$，$n−1$ 转化成二进制便全由1组成，令 `x^(n-1)` ，即可使得 $x$ 按位取反，求得 $c(x)$ 

再分析剩下两种情况。

当 $0＜k＜n−1$ 时，使 0 和 $c(k)$ 配对，$k$ 和 $n-1$ 配对即可。

当 $k=n-1$ 时，若 $n=4$ 显然输出 -1；其余情况，使 $0$ 和 $2$ 配，$n-1$ 和 $n-2$ 配，$n-3$ 和 $1$ 配即可。

#### 总结

对于构造类题目，可以先尝试思考特殊情况，说不定能推广到一般。

#### AC代码

|Problem|Lang|Verdict|Time|Memory|
|:--:|:--:|:--:|:--:|:--:|
|[C - And Matching](https://codeforces.com/contest/1631/problem/C)|GNU C++17|**Accepted**|108 ms|0 KB|

```cpp
#include<bits/stdc++.h>
using namespace std;
int n;

int c(int x) 
{
    return x ^ (n - 1);
}

void solve()
{
    int k;
    cin >> n >> k;
    if (k == 0) {
        for (int i = 0; i <= n / 2 - 1; i++)
            cout << i << ' ' << c(i) << endl;
    }
    else if (k == n - 1) {
        if (n == 4) {
            cout << -1 << endl;
        }
        else {
            cout << n - 1 << ' ' << n - 2 << endl;
            cout << n - 3 << ' ' << 1 << endl;
            cout << 0 << ' ' << 2 << endl;
            for (int i = 3; i <= n / 2-1; i++)
                cout << i << ' ' << c(i) << endl;
        }
    }
    else {
        cout << 0 << ' ' << c(k) << endl;
        cout << k << ' ' << n - 1 << endl;
        for (int i = 1; i <= n / 2 - 1; i++)
            if(i!=k&&i!=c(k))cout << i << ' ' << c(i) << endl;
    }
}

int main()
{
    int tt;
    cin >> tt;
    while (tt--) {
        solve();
    }
    return 0;
}
```

---

##### by syj

//希望哪天div2能在场上出一道c啊。

//成功！希望出次d

## Codeforces（Div.2）  补题 合集1

### C. Meximum Array

#### 题意

共 $t$ $(1\leq t\leq 100)$组测试，给出数组元素 $n$ $(1≤n≤2⋅10^5)$ ，随后给出 $n$ 个数，$0≤a_i≤n$

可以做如下操作：

选择前 $k$ 个数，求出它们的 $MEX$ ，放在 $b$ 数组的末尾，同时删去前 $k$ 个数。

要求让 $b$ 数组字典序上最大，输出 $b$ 的长度和组成。

#### 思路

贪心思想，因为是按字典序，所以越早存入 $b$ 的应该越大，即每次找出最大的mex存入，并在次情况下要求删除最少的数字。对于求mex，可以先在输入的时候，预处理记录每个数字出现的次数，保存在另一个数组中，随后遍历该数组，找出第一个等于0的元素即可。但由于 $n$ 的值较大，所以每次都遍历会超时。

可以发现，对于有相同部分的两段数组元素的mex值是有关系的。假如不同部分有等于mex的值，mex就应该加1……当然也可以想到有很多不同的情况。但可以先确定储存更新mex的策略。

如何更新mex值呢，可以在遍历 $a$ 的循环里加个循环，即如果在已搜索过的 $a[i]$ 里出现了mex，那就将mex++，再看新的mex有没有……而当之后未搜索的 $a[i]$ 里没有等于mex的，该整个数组的mex就已经求出了。（由于预处理了每个数的个数，那么每搜索一个就让该数的次数减一，如果mex的剩余次数为0，就代表mex已经求出）。

#### 总结

学习到了复杂度为 $nlog(n)$ 的复杂度的求 $mex$ 的方法。其实这也应该是我们用人脑去考虑 $mex$ 时的做法（至少是我的）。所以有时我们本身的思维方式也可以带来程序设计的灵感。

> 要重视预处理的作用，可以简化很多运算。

#### AC代码

|Problem|Lang|Verdict|Time|Memory|
|:--:|:--:|:--:|:--:|:--:|
|[C - Meximum Array](https://codeforces.com/contest/1629/problem/C)|GNU C++17|**Accepted**|46 ms|3100 KB|

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=200010;
int a[N];
int ct[N];//该数字剩余次数
int r[N];//a中的第几个数字使用在第几轮
int b[N];
int n;
void solve()
{
    cin >> n;
    memset(ct, 0, sizeof(ct));
    memset(b, 0, sizeof(b));
    memset(r, 0, sizeof(r));
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
        ct[a[i]]++;
    }
    int id = 1;
    int m = 0;
    for (int i = 1; i <= n; i++) {
        ct[a[i]]--;
        r[a[i]] = id;//代表a[i]这个数存在且在第一轮
        while (r[m] == id)
            m++;
        if (ct[m] == 0) //说明mex已经是最大的了
        {
            b[id] = m;
            id++;
            m = 0;
        }
    }
    cout << id-1 << endl;
    for (int i = 1; i <= id-1; i++)
        cout << b[i] << ' ';
    cout << endl;
}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    int tt;
    cin >> tt;
    while (tt--) {
        solve();
    }
    return 0;
}
```

---

### C. Poisoned Dagger

#### 题意

$t$ $(1≤t≤1000)$ 组测试，给出 $n(1≤n≤100)$ 和 $h(1≤h≤10^{18})$ ，分别表示数组元素个数和怪物血量，对于各数组元素，$a_i(1≤a_i≤10^9)$  。数组元素的值表示每次攻击开始的时刻，持续 $k$ 个单位时间（从该时刻开始），每个单位时间对怪物造成1点伤害，直到下一次攻击停止，更新攻击效果。

求能使得怪物被打败的最小的$k$ 。

#### 思路

由于 $h$ 的数值很大，所以不能采取逐个累加的方式，会超时；所以使用二分查找 $k$ ，通过一个函数来判断是否可以，因为 $n$ 不大，所以可以通过遍历数组 $a$ 来判断即可。

#### 总结

> 二分查找（Binary search）是个好东西。

#### AC代码

|Problem|Lang|Verdict|Time|Memory|
|:--:|:--:|:--:|:--:|:--:|
|[C - Poisoned Dagger](https://codeforces.com/contest/1613/problem/C)|GNU C++17|**Accepted**|46 ms|0 KB|

```cpp
#include<bits/stdc++.h>
using namespace std;
long long n, h;
long long a[110];

bool check(long long x)
{
    long long sum = 0;
    for (int i = 1; i < n; i++){
        sum += min(x, a[i + 1] - a[i]);
    }
    sum += x;
    return sum >= h;
}
void solve()
{
    cin >> n >> h;
    for (int i = 1; i <= n; i++)
        cin >> a[i];
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

}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    int tt;
    cin >> tt;
    while (tt--) {
        solve();
    }
    return 0;
}
```

---

### C. Chat Ban

#### 题意

共 $t$ $(1≤t≤10^4)$组数据，给出 $k$ 和 $x$ $(1≤k≤10^9;1≤x≤10^{18})$ ，从第一行到第 $k$ 行每行的个数为行数，从 $k+1$ 行递减至 $0$ ,总个数需要小于等于 $x$ ，问最多可以有几行（包括不完整的行）。

#### 思路

因为 $x$ 较大，所以采用二分查找，将整个图形分成上下两个三角形即可。二分的主要麻烦的地方就是端点的调整。

#### 总结

> Binary search yyds！！！

#### AC代码

|Problem|Lang|Verdict|Time|Memory|
|:--:|:--:|:--:|:--:|:--:|
|[C - Chat Ban](https://codeforces.com/contest/1612/problem/C)|GNU C++17|**Accepted**|31 ms|0 KB|

```cpp
#include<bits/stdc++.h>
using namespace std;
long long k, x;
long long ch1(long long pos)
{
    if ((2 + pos) * (pos+1) / 2>=x)
        return 1;
    else return 0;
}
long long ch2(long long pos)
{
    if ((k-1+k-pos)*pos/2>=x)
        return 1;
    else return 0;
}
void solve()
{
    cin >> k >> x;
    long long sum = 0;
    if ((1 + k) * k / 2 >= x) {
        long long l = 0, r = k;
        long long ans = 0;
        while (l <= r)
        {
            long long  mid = (l + r) / 2;
            if (ch1(mid))
            {
                r = mid - 1;
                ans = mid;
            }
            else
            {
                l = mid + 1;
            }
        }
        cout << ans + 1 << endl;
        return;
    }
    else if ((1 + k) * k / 2 + (1 + k - 1) * (k - 1) / 2 <= x)
        cout << 2 * k - 1 << endl;
    else {
        x -= (1 + k) * k / 2;
        long long l = 1, r = k;
        long long ans = 0;
        while (l <= r)
        {
            long long mid = (l + r) / 2;
            if (ch2(mid))
            {
                r = mid - 1;
                ans = mid;
            }
            else
            {
                l = mid + 1;
            }
        }
        cout<<k+ans<<endl;
    }
}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    int tt;
    cin >> tt;
    while (tt--) {
        solve();
    }
    return 0;
}
```

---

### B. Special Permutation

#### 题意

给出 $n,a,b$ ，分别为数组元素个数，所求排列的左半最小值和右半最大值，排列由 $1$ 到 $n$ 组成 。如果能做到，输出排列；不然输出-1。

#### 思路

先分类存储，均可和已固定位置的和均不可的，随后分类讨论即可。

#### 总结

做的时候相当然了，看到 $n≤100$ ，就给左右的数组各开了60，没想到运算过程中越界了。

> 所以不仅看输入输出，也要考虑运算过程中是否存在越界。

#### AC代码

|Problem|Lang|Verdict|Time|Memory|
|:--:|:--:|:--:|:--:|:--:|
|[B - Special Permutation](https://codeforces.com/contest/1612/problem/B)|GNU C++17|**Accepted**|15 ms|0 KB|

```cpp
#include<bits/stdc++.h>
using namespace std;
int n, a, b;
int l[100];
int r[100];
int m[100];
//比a大可以放左边，比b小可以放右边
void solve()
{
    cin >> n >> a >> b;
    l[1] = a;
    r[1] = b;
    int re = 0;
    int cnt1 = 1;
    int cnt2 = 1;
    for (int i = 1; i <= n; i++) {
        if (i == a || i == b)continue;
        if (i > a && i < b) {
            re++;
            m[re] = i;
        }
        else if (i > a && i > b) {
            cnt1++;
            l[cnt1] = i;//报错：out of bounds
            //如果l，r数组没开到100，这里的cnt可能会越界
        }
        else if(i < a && i < b){
            cnt2++;
            r[cnt2] = i;
        }
        else if (i < a && i > b) {
            cout << -1 << endl;
            return;
        }
    }
    if (abs(cnt2 - cnt1) > re)cout << -1 << endl;
    else {
        while (cnt1 != cnt2||re) {
            if (cnt1 < cnt2) {
                cnt1++;
                l[cnt1] = m[re];
                re--;
            }
            else if(cnt1>cnt2){
                cnt2++;
                r[cnt2] = m[re];
                re--;
            }
            else {
                while (re) {
                    cnt1++;
                    l[cnt1] = m[re];
                    re--;
                    cnt2++;
                    r[cnt2] = m[re];
                    re--;
                }
            }
        }
        for (int i = 1; i <= cnt1; i++)
            cout << l[i] << ' ';
        for (int i = 1; i <= cnt2; i++)
            cout << r[i] << ' ';
        cout << endl;
    }
}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    int tt;
    cin >> tt;
    while (tt--) {
        solve();
    }
    return 0;
}
```

---

### D. Make Them Equal

#### 题意

给出各 $n$ 个元素的数组 $b,c$  ，以及最多操作数 $k$ 。

初始时另一个数组 $a$ 中每个元素都是 $1$ ，每次操作可以使 $a_i=a_i+⌊a_i/x⌋$ 

`if(a[i]==b[i])sum+=c[i]`，求 $sum$ 的最大值。

数据范围：

$1≤i≤n,x>0,1≤t≤100,1≤n≤10^3;0≤k≤10^6,1≤b_i≤10^3,1≤c_i≤10^6$

#### 思路

dp（背包）。

先打表出到达1到1000所需要的步数，再对于每次加不加该操作判断即可。

注意到 $k$ 的数据较大，所有1变成1000，即使总共1000个数，也只需要12000步，所以开始时对 $k$ 处理一下，使它对 12000 取 min ，这样一来dp数组大小也只需要取12010左右，同时运算量减少很多（不然本题会tle）。 

#### 总结

当发现变量范围很大时，看看能不能剪枝，以免简单大数据超时。

#### AC代码

|Problem|Lang|Verdict|Time|Memory|
|:--:|:--:|:--:|:--:|:--:|
|[D - Make Them Equal](https://codeforces.com/contest/1633/problem/D)|GNU C++17|**Accepted**|31 ms|100 KB|

```cpp
#include<bits/stdc++.h>
using namespace std;
int ti[2010];
int b[2010];
int c[2010];
int dp[12010];
int n, k;
void init()
{
    memset(ti, 0x3f, sizeof(ti));//这里注意，因为是取min，初始化成最大值
    ti[1] = 0;//起点要给出
    for (int i = 1; i < 1000; i++)
        for (int j = 1; j <= i; j++)
            ti[i + i / j] = min(ti[i] + 1, ti[i + i / j]);
}
void solve()
{
    cin >> n >> k;
    k = min(k, 12 * n);//剪枝，ti[1000]=12
    memset(dp, 0, sizeof(dp));
    for (int i = 1; i <= n; i++)
        cin >> b[i];
    for (int i = 1; i <= n; i++)
        cin >> c[i];
    for (int i = 1; i <= n; i++)
        for (int j = k; j >= ti[b[i]]; j--)
            dp[j] = max(dp[j], dp[j - ti[b[i]]] + c[i]);
    cout << dp[k] << endl;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    int tt;
    cin>>tt;
    init();
    while (tt--)solve();
    return 0;
}
```

---
