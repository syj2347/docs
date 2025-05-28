# 🪝Leetcode_3

### Easy

### Medium

##### [连接两棵树后最大目标节点数目 I](https://leetcode.cn/problems/maximize-the-number-of-target-nodes-after-connecting-trees-i/description/ "https://leetcode.cn/problems/maximize-the-number-of-target-nodes-after-connecting-trees-i/description/")

​`5.28 dfs (自己解法超时) 这段时间做题做得少，太生疏了，就算超时的做法也debug了好久`​

```cpp
// 需要使用邻接表，使用set<pair<int,int>>存边超时了
/*
 * @lc app=leetcode.cn id=3372 lang=cpp
 *
 * [3372] 连接两棵树后最大目标节点数目 I
 */

// @lc code=start
class Solution
{
public:
    vector< vector< int > > init(vector< vector< int > > &e)
    {
        int n = e.size() + 1;
        vector< vector< int > > g(n);
        for (auto x : e)
            g[x[0]].push_back(x[1]), g[x[1]].push_back(x[0]);
        return g;
    }
    int dfs(vector< vector< int > > &a, int last, int cnt, int kk, int st)
    {
        if (cnt > kk)
            return 0;
        int ans = 1;
        for (auto x : a[st])
        {
            if (last != x)
                ans += dfs(a, st, cnt + 1, kk, x);
        }
        return ans;
    }
    vector< int > maxTargetNodes(vector< vector< int > > &edges1,
                                 vector< vector< int > > &edges2, int k)
    {
        int n = edges1.size() + 1, m = edges2.size() + 1;
        vector< vector< int > > vt1 = init(edges1), vt2 = init(edges2);
        int mx2 = 0;
        for (int i = 0; i < m; i++)
            mx2 = max(mx2, dfs(vt2, i, 0, k - 1, i));
        vector< int > res(n, 0);
        for (int i = 0; i < n; i++)
            res[i] = dfs(vt1, i, 0, k, i) + mx2;
        return res;
    }
};
// @lc code=end

```

```cpp
// 原解答：超时
class Solution
{
public:
    int kk;
    void dfs(set< pair< int, int > > &a, vector< int > &vis, int &n, int st,
             int &end, int &cost, bool &flag)
    {
        if (flag == true || cost > kk)
            return;
        if (st == end)
        {
            flag = true;
            return;
        }
        for (int i = 0; i < n; i++)
        {
            if (!vis[i] && a.find(make_pair(st, i)) != a.end())
            {
                cost++, vis[i] = 1;
                dfs(a, vis, n, i, end, cost, flag);
                if (flag == true)
                    return;
                cost--, vis[i] = 0;
            }
        }
        return;
    }
    vector< int > cal(set< pair< int, int > > &a, int st, int sz)
    {
        vector< int > ans(sz, -1);
        for (int i = 0; i < sz; i++)
        {
            if (i == st)
            {
                ans[i] = 0;
                continue;
            }
            vector< int > vis(sz, 0);
            int cost = 0;
            bool flag = false;
            vis[st] = 1;
            dfs(a, vis, sz, st, i, cost, flag);
            if (flag == true)
                ans[i] = cost;
        }
        return ans;
    }
    vector< int > maxTargetNodes(vector< vector< int > > &edges1,
                                 vector< vector< int > > &edges2, int k)
    {
        int n = edges1.size() + 1, m = edges2.size() + 1;
        kk = k;
        set< pair< int, int > > set1, set2;
        for (int i = 0; i < n - 1; i++)
            set1.insert(make_pair(edges1[i][0], edges1[i][1])),
                set1.insert(make_pair(edges1[i][1], edges1[i][0]));
        for (int i = 0; i < m - 1; i++)
            set2.insert(make_pair(edges2[i][0], edges2[i][1])),
                set2.insert(make_pair(edges2[i][1], edges2[i][0]));
        vector< vector< int > > dis1, dis2;
        for (int i = 0; i < n; i++)
            dis1.push_back(cal(set1, i, n));
        for (int i = 0; i < m; i++)
            dis2.push_back(cal(set2, i, m));
        vector< int > ans(n, 0);
        for (int i = 0; i < n; i++)
        {
            int j = i;
            for (int p = 0; p < m; p++)
            {
                // connect j p
                int cnt = 0;
                for (int l = 0; l < m; l++)
                {
                    if (dis1[i][j] != -1 && dis2[p][l] != -1 &&
                        dis1[i][j] + dis2[p][l] + 1 <= k)
                        cnt++;
                }
                for (int l = 0; l < n; l++)
                {
                    if (dis1[i][l] != -1)
                        cnt++;
                }
                ans[i] = max(ans[i], cnt);
            }
        }
        return ans;
    }
};
```

##### [零数组变换 III](https://leetcode.cn/problems/zero-array-transformation-iii/description/ "https://leetcode.cn/problems/zero-array-transformation-iii/description/")

​`5.22 贪心+优先队列+差分数组 （看题解才会的）`​

```cpp
/*
 * @lc app=leetcode.cn id=3362 lang=cpp
 *
 * [3362] 零数组变换 III
 */

// @lc code=start
class Solution
{
public:
    int maxRemoval(vector< int > &nums, vector< vector< int > > &queries)
    {
        int n = nums.size(), j = 0, s = 0;
        vector< int > diff(n + 1, 0);
        priority_queue< int > pq;
        ranges::sort(queries, {},
                     [](auto &q)
                     {
                         return q[0];
                     });
        for (int i = 0; i < n; i++)
        {
            s += diff[i];
            while (j < queries.size() && queries[j][0] <= i)
            {
                pq.push(queries[j][1]);
                j++;
            }
            while (s < nums[i] && !pq.empty() && pq.top() >= i)
            {
                s++;
                diff[pq.top() + 1]--;
                pq.pop();
            }
            if (s < nums[i])
                return -1;
        }
        return pq.size();
    }
};
// @lc code=end
```

##### [零数组变换 II](https://leetcode.cn/problems/zero-array-transformation-ii/description/ "https://leetcode.cn/problems/zero-array-transformation-ii/description/")

​`5.21 二分+差分`​

```cpp
/*
 * @lc app=leetcode.cn id=3356 lang=cpp
 *
 * [3356] 零数组变换 II
 */

// @lc code=start
class Solution
{
public:
    bool check(vector< int > &nums, vector< vector< int > > &queries, int _)
    {
        int n = nums.size();
        vector< int > cnt(n + 1, 0);
        for (int i = 0; i < _; i++)
        {
            int l = queries[i][0], r = queries[i][1], v = queries[i][2];
            cnt[l] += v;
            cnt[r + 1] -= v;
        }
        bool flag = true;
        int c = 0;
        for (int i = 0; i < n; i++)
        {
            c += cnt[i];
            if (nums[i] > c)
            {
                flag = false;
                break;
            }
        }
        return flag;
    }
    int minZeroArray(vector< int > &nums, vector< vector< int > > &queries)
    {
        int l = 0, r = queries.size();
        int ans = -1;
        while (l <= r)
        {
            int m = l + (r - l) / 2;
            if (check(nums, queries, m) == true)
            {
                r = m - 1, ans = m;
            } else
                l = m + 1;
        }
        return ans;
    }
};
// @lc code=end
```

##### [零数组变换 I](https://leetcode.cn/problems/zero-array-transformation-i/)

​`5.20 使用了树状数组的板子；可以使用差分数组`​

```cpp
class Solution {
public:
    int n;
    int c[100010];
    void add(int p, int x) {
        while (p <= n)
            c[p] += x, p += p & -p;
    }
    void range_add(int l, int r, int x) { add(l, x), add(r + 1, -x); }
    int ask(int p) {
        int res = 0;
        while (p)
            res += c[p], p -= p & -p;
        return res;
    }
    bool isZeroArray(vector<int>& nums, vector<vector<int>>& queries) {
        n = nums.size();
        memset(c, 0, sizeof(c));
        for (int i = 0; i < queries.size(); i++) {
            int l = queries[i][0], r = queries[i][1];
            range_add(l + 1, r + 1, 1);
        }
        bool flag = true;
        for (int i = 0; i < n; i++) {
            if (nums[i] > ask(i + 1)) {
                flag = false;
                break;
            }
        }
        return flag;
    }
};
```

```cpp
/*
 * @lc app=leetcode.cn id=3355 lang=cpp
 *
 * [3355] 零数组变换 I
 */

// @lc code=start
class Solution
{
public:
    bool isZeroArray(vector< int > &nums, vector< vector< int > > &queries)
    {
        int n = nums.size();
        vector< int > cnt(n + 1, 0);
        for (auto x : queries)
        {
            int l = x[0], r = x[1];
            cnt[l] += 1;
            cnt[r + 1] -= 1;
        }
        int c = 0;
        bool flag = true;
        for (int i = 0; i < n; i++)
        {
            c += cnt[i];
            if (c < nums[i])
            {
                flag = false;
                break;
            }
        }
        return flag;
    }
};
// @lc code=end
```

### Hard
