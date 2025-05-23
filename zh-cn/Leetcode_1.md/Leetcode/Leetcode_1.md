# Leetcode_1

### Easy

- [X] ##### [找出所有子集的异或总和再求和](https://leetcode.cn/problems/sum-of-all-subset-xor-totals/description/?envType=daily-question&amp;envId=2025-04-05)

  ​`4.5 多种解法 递归法 迭代法 数学`​

- [X] ##### [有序三元组中的最大值 I](https://leetcode.cn/problems/maximum-value-of-an-ordered-triplet-i/description/)

  - [X] [有序三元组中的最大值 II](https://leetcode.cn/problems/maximum-value-of-an-ordered-triplet-ii/description/?envType=daily-question&amp;envId=2025-04-03)  `4.3 数据加强版`​

  ​`4.2 多种解法的思路比较值得记忆`​

```cpp
// 维护前后缀数组
class Solution
{
public:
    long long maximumTripletValue(vector< int > &nums)
    {
        int len = nums.size();
        vector< int > lmax(len, 0), rmax(len, 0);
        lmax[0] = nums[0], rmax[0] = nums[len - 1];
        for (int i = 1; i < len; i++)
        {
            lmax[i] = max(lmax[i - 1], nums[i]);
            rmax[i] = max(rmax[i - 1], nums[len - i - 1]);
        }
        long long ans = 0;
        for (int j = 1; j < len - 1; j++)
        {
            long long x = lmax[j - 1] - nums[j];
            long long y = rmax[len - j - 2];
            ans = max(ans, x * y);
        }
        return ans;
    }
};
```

### Medium

- [X] ##### [分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/description/?envType=daily-question&amp;envId=2025-04-07)

  ​`4.7 经典01背包 复习了下背包问题 写出来了`​

  ```cpp
  // @lc code=start
  // 4.7 每日一题
  class Solution
  {
  public:
      bool canPartition(vector< int > &nums)
      {
          int sum = 0;
          for (auto x : nums)
              sum += x;
          if (sum % 2)
              return false;
          int n = nums.size();
          sum /= 2;
          vector< vector< int > > dp(n, vector< int >(sum + 1, 0));
          dp[0][0] = 1;
          if (nums[0] <= sum)
              dp[0][nums[0]] = 1;
          for (int i = 1; i < n; i++)
          {
              for (int j = 0; j <= sum; j++)
              {
                  dp[i][j] =
                      dp[i - 1][j] || j >= nums[i] && dp[i - 1][j - nums[i]];
              }
          }
          return dp[n - 1][sum] == 1;
      }
  };
  // @lc code=end
  ```

- [X] ##### [最大整除子集](https://leetcode.cn/problems/largest-divisible-subset/description/?envType=daily-question&amp;envId=2025-04-06)

  ​`4.6 自己用了两个方法都超时了，dfs是39/49，维护出所有集合的方法是47/49处超时`​

  ​`看了官方题解的一个思路（如果是最大的数的倍数可以插入）后，去试着自己写了下，还是超时 47/49`​

  ```cpp
  // 看完题解后写的，动态规划
  class Solution
  {
  public:
      vector< int > largestDivisibleSubset(vector< int > &nums)
      {
          sort(nums.begin(), nums.end());
          int len = nums.size();
          int mx = 1;
          vector< int > dp(len, 1);
          for (int i = 1; i < len; i++)
          {
              for (int j = 0; j < i; j++)
              {
                  if (nums[i] % nums[j] == 0)
                      dp[i] = max(dp[i], dp[j] + 1);
              }
              mx = max(mx, dp[i]);
          }
          vector< int > ans;
          int tmx = mx;
          int last;
          for (int i = len - 1; i >= 0; i--)
          {
              if (dp[i] == tmx)
              {
                  if (mx == tmx || last % nums[i] == 0)
                  {
                      ans.push_back(nums[i]);
                      tmx -= 1;
                      last = nums[i];
                  }
              }
          }
          return ans;
      }
  };
  ```

- [X] ##### [解决智力问题](https://leetcode.cn/problems/solving-questions-with-brainpower/description/?envType=daily-question&amp;envId=2025-04-01)  

  <span data-type="text" style="color: var(--b3-theme-on-background); font-family: var(--b3-font-family-protyle); font-size: var(--b3-font-size-editor); background-color: transparent;"> </span>`4.1`​

  ```cpp
  // 自己的解法，思路来源于前缀和（打表法）
  typedef long long ll;
  class Solution
  {
  public:
      long long mostPoints(vector< vector< int > > &questions)
      {
          int len = questions.size();
          vector< ll > amn(len, 0);
          ll mx = 0;
          ll amx = 0;
          for (int i = 0; i < len; i++)
          {
              ll p = questions[i][0];
              ll b = questions[i][1];
              amx = max(amx, amn[i]);
              mx = max(mx, p + amx);
              int nxt = i + b + 1;
              if (nxt < len)
                  amn[nxt] = max(amn[nxt], p + amx);
          }
          return mx;
      }
  };
  // 官方题解：倒序dp √
  // 记忆化dfs
  ```

### Hard

- [X] ##### [统计好整数的数目](https://leetcode.cn/problems/find-the-count-of-good-integers/)

  ​`4.12`​

  ```cpp
  // @lc code=start
  // 4.12 每日一题 看题解后写的 枚举+排列组合
  typedef long long ll;
  class Solution
  {
  public:
      long long countGoodIntegers(int n, int k)
      {
          unordered_set< string > st;
          int b = pow(10, (n - 1) / 2);
          int skip = n % 2;
          for (int i = b; i < b * 10; i++)
          {
              string s = to_string(i);
              s += string(s.rbegin() + skip, s.rend());
              ll num = stoll(s);
              if (num % k == 0)
              {
                  sort(s.begin(), s.end());
                  st.insert(s);
              }
          }
          vector< ll > f(n + 1, 1);
          for (int i = 1; i <= n; i++)
              f[i] = f[i - 1] * i;
          int ans = 0;
          for (auto s : st)
          {
              vector< int > cnt(10);
              for (auto c : s)
                  cnt[c - '0']++;
              ll p = (n - cnt[0]) * f[n - 1];
              for (auto x : cnt)
                  p /= f[x];
              ans += p;
          }
          return ans;
      }
  };
  // @lc code=end
  ```

- [X] ##### [统计强大整数的数目](https://leetcode.cn/problems/count-the-number-of-powerful-integers/)

  ​`4.10`

  ```cpp
  // @lc code=start
  // 4.10 每日一题
  // 看了题解写的 数形dp （需要复习）
  class Solution
  {
  public:
      long long f[20];
      long long solve(string num, bool flag, int id, int limit, string s)
      {
          int nlen = num.size(), slen = s.size();
          if (nlen < slen)
              return 0;
          if (!flag && f[id] != -1)
              return f[id];
          if (nlen - id == slen)
          {
              if (flag)
                  return s <= num.substr(id);
              else
                  return 1;
          }
          long long ans = 0;
          int up = min(limit, flag ? num[id] - '0' : 9);
          for (int i = 0; i <= up; i++)
              ans += solve(num, flag && i == (num[id] - '0'), id + 1, limit, s);
          if (!flag)
              f[id] = ans;
          return ans;
      }
      long long numberOfPowerfulInt(long long start, long long finish, int limit,
                                    string s)
      {
          memset(f, -1, sizeof(f));
          long long a = solve(to_string(start - 1), true, 0, limit, s);
          memset(f, -1, sizeof(f));
          long long b = solve(to_string(finish), true, 0, limit, s);
          return b - a;
      }
  };
  // @lc code=end
  ```

- [X] ##### [串联所有单词的子串](https://leetcode.cn/problems/substring-with-concatenation-of-all-words/description/?envType=study-plan-v2&amp;envId=top-interview-150) 

   `3.31`​

  ```cpp
  // 3.31
  class Solution
  {
  public:
      vector< int > findSubstring(string s, vector< string > &words)
      {
          vector< int > ans;
          int wlen = words.size();
          int osize = words[0].size();
          int wsize = wlen * osize;
          int slen = s.size();
          unordered_map< string, int > mp;
          for (int i = 0; i < wlen; i++)
              mp[words[i]] += 1;
          unordered_set< string > okset;
          for (int i = 0; i + wsize - 1 < slen; i++)
          {
              // 面向测试样例编程，给我整过了，不过感觉还是不太对
              if (okset.find(s.substr(i, wsize)) != okset.end())
              {
                  ans.push_back(i);
                  continue;
              }
              int flag = 0;
              int j = i;
              unordered_map< string, int > tmp = mp;
              while (j < slen)
              {
                  string ss = s.substr(j, osize);
                  if (tmp.find(ss) != tmp.end() && tmp[ss] > 0)
                      tmp[ss]--;
                  else
                      break;
                  j += osize;
                  if (j - i == wsize)
                  {
                      flag = 1;
                      break;
                  }
              }
              if (flag)
              {
                  ans.push_back(i);
                  okset.insert(s.substr(i, wsize));
              }
          }
          return ans;
      }
  };
  ```

  ```cpp
  // 看了官方题解思路，滑动窗口拿下
  class Solution
  {
  public:
      vector< int > findSubstring(string s, vector< string > &words)
      {
          vector< int > ans;
          int wlen = words.size();
          int osize = words[0].size();
          int wsize = wlen * osize;
          int slen = s.size();
          unordered_map< string, int > mp;
          for (int i = 0; i < wlen; i++)
              mp[words[i]] += 1;
          for (int i = 0; i < osize && i + wsize - 1 < slen; i++)
          {
              unordered_map< string, int > tmp;
              for (int j = i; j <= i + wsize - 1; j += osize)
              {
                  string ss = s.substr(j, osize);
                  tmp[ss]++;
              }
              for (int j = i; j + wsize - 1 < slen; j += osize)
              {
                  if (tmp == mp)
                      ans.push_back(j);
                  string p = s.substr(j, osize);
                  if (--tmp[p] == 0)
                      tmp.erase(p);
                  tmp[s.substr(j + wsize, osize)]++;
              }
          }
          return ans;
      }
  };
  ```

- [X] ##### [最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/description/?envType=study-plan-v2&amp;envId=top-interview-150) 

   `3.31`​

  ```cpp
  // 3.31 滑动窗口
  // ps：感觉比上次写得优美
  class Solution
  {
  public:
      string minWindow(string s, string t)
      {
          int slen = s.size(), tlen = t.size();
          unordered_map< char, int > mp;
          unordered_set< char > st;
          for (auto c : t)
          {
              mp[c]++;
              st.insert(c);
          }
          string ans;
          int mn = slen + 1;
          unordered_map< char, int > tmp;
          unordered_set< char > okset;
          int i = 0, j = 0;
          while (i + tlen - 1 < slen && j < slen)
          {
              if (mp.find(s[j]) != mp.end() && ++tmp[s[j]] == mp[s[j]])
              {
                  okset.insert(s[j]);
                  if (okset.size() == st.size())
                  {
                      while (i + tlen - 1 < slen)
                      {
                          if (mp.find(s[i]) == mp.end())
                              i++;
                          else if (tmp[s[i]] > mp[s[i]])
                              tmp[s[i]]--, i++;
                          else
                              break;
                      }
                      if (j - i + 1 < mn)
                      {
                          mn = j - i + 1;
                          ans = s.substr(i, mn);
                      }
                      tmp[s[i]]--, okset.erase(s[i]);
                      i++, j++;
                      continue;
                  }
              }
              j++;
          }
          if (mn == slen + 1)
              return string("");
          else
              return ans;
      }
  };
  ```
