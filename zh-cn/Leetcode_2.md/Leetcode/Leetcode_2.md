# Leetcode_2

### Easy

### Medium

##### [字符串转换后的长度 I](https://leetcode.cn/problems/total-characters-in-string-after-transformations-i/)

  ​`5.13 写了两种方法，使用map或者unordered_map时都超时了，注意当数据较大时可以使用数组`​

  ```cpp
  /*
   * @lc app=leetcode.cn id=3335 lang=cpp
   *
   * [3335] 字符串转换后的长度 I
   */

  // @lc code=start
  const int d = 1e9 + 7;
  typedef long long ll;
  class Solution
  {
  public:
      int solve(char c, int t, vector< vector< int > > &memo)
      {
          if (memo[c - 'a'][t])
              return memo[c - 'a'][t];
          ll p = 'z' - c + 1;
          if (t >= p)
              memo[c - 'a'][t] =
                  (solve('a', t - p, memo) % d + solve('b', t - p, memo) % d) % d;
          else
              memo[c - 'a'][t] = 1;
          return memo[c - 'a'][t];
      }
      int lengthAfterTransformations(string s, int t)
      {
          vector< vector< int > > memo(26, vector< int >(t + 1, 0));
          unordered_map< char, ll > mp;
          for (auto x : s)
              mp[x]++;
          ll ans = 0;
          for (auto x : mp)
              ans = (ans + (x.second * solve(x.first, t, memo) % d)) % d;
          return ans;
      }
  };
  // @lc code=end
  ```

##### [到达最后一个房间的最少时间 II](https://leetcode.cn/problems/find-minimum-time-to-reach-last-room-ii/description/ "https://leetcode.cn/problems/find-minimum-time-to-reach-last-room-ii/description/")

  ​`5.8 dij 不使用priority_queue会超时；dij不熟练，需要复习`​

  ```cpp
  /*
   * @lc app=leetcode.cn id=3342 lang=cpp
   *
   * [3342] 到达最后一个房间的最少时间 II
   */

  // @lc code=start
  // 5.8 每日一题
  class Solution
  {
  public:
      int dir[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
      struct Node
      {
          int x;
          int y;
          int dis;
          Node(int x, int y, int dis) : x(x), y(y), dis(dis) {}
          bool operator<(const Node &p) const
          {
              return dis > p.dis;
          }
      };
      int minTimeToReach(vector< vector< int > > &moveTime)
      {
          priority_queue< Node > q;
          int n = moveTime.size(), m = moveTime[0].size();
          vector< vector< int > > dist(n, vector< int >(m, INT_MAX));
          dist[0][0] = 0;
          q.push(Node(0, 0, 0));
          while (!q.empty())
          {
              Node tmp = q.top();
              q.pop();
              int tx = tmp.x, ty = tmp.y;
              for (int i = 0; i < 4; i++)
              {
                  int nx = tx + dir[i][0];
                  int ny = ty + dir[i][1];
                  if (nx < 0 || nx >= n || ny < 0 || ny >= m)
                      continue;
                  int p = (tx + ty) % 2 + 1;
                  int dmx = max(dist[tx][ty], moveTime[nx][ny]) + p;
                  if (dmx < dist[nx][ny])
                  {
                      dist[nx][ny] = dmx;
                      q.push(Node(nx, ny, dmx));
                  }
              }
          }
          return dist[n - 1][m - 1];
      }
  };
  // @lc code=end
  ```

##### [到达最后一个房间的最少时间 I](https://leetcode.cn/problems/find-minimum-time-to-reach-last-room-i/description/ "https://leetcode.cn/problems/find-minimum-time-to-reach-last-room-i/description/")

  ​`5.7 bfs||dij 看了题解才做出，自己的方法超时了；bfs需要剪枝`​

  ```cpp
  /*
   * @lc app=leetcode.cn id=3341 lang=cpp
   *
   * [3341] 到达最后一个房间的最少时间 I
   */

  // @lc code=start
  // 5.7 每日一题
  class Solution
  {
  public:
      int ans;
      int dir[5][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
      int minTimeToReach(vector< vector< int > > &moveTime)
      {
          queue< pair< int, int > > q;
          int n = moveTime.size(), m = moveTime[0].size();
          vector< vector< int > > dist(n, vector< int >(m, INT_MAX));
          dist[0][0] = 0;
          q.push(make_pair(0, 0));
          while (!q.empty())
          {
              pair< int, int > tmp = q.front();
              q.pop();
              int tx = tmp.first, ty = tmp.second;
              for (int i = 0; i < 4; i++)
              {
                  int nx = tx + dir[i][0];
                  int ny = ty + dir[i][1];
                  if (nx < 0 || nx >= n || ny < 0 || ny >= m)
                      continue;
                  if (dist[nx][ny] <= max(dist[tx][ty] + 1, moveTime[nx][ny] + 1))
                      continue;
                  q.push(make_pair(nx, ny));
                  dist[nx][ny] = max(dist[tx][ty] + 1, moveTime[nx][ny] + 1);
              }
          }
          return dist[n - 1][m - 1];
      }
  };
  // @lc code=end
  ```

##### [多米诺和托米诺平铺](https://leetcode.cn/problems/domino-and-tromino-tiling/)

  ​`5.5 看题解才做出来的 对动态规划的状态转移的敏感度不够`​

  ```cpp
  typedef long long ll;
  class Solution {
  public:
      const ll d = 1e9 + 7;
      int numTilings(int n) {
          vector<vector<ll>> f(n + 1, vector<ll>(4, 0));
          f[0][3] = 1;
          for (int i = 1; i <= n; i++) {
              f[i][0] = f[i - 1][3];
              f[i][1] = (f[i - 1][0] + f[i - 1][2]) % d;
              f[i][2] = (f[i - 1][0] + f[i - 1][1]) % d;
              f[i][3] =
                  (f[i - 1][0] + f[i - 1][1] + f[i - 1][2] + f[i - 1][3]) % d;
          }
          return f[n][3];
      }
  };
  ```

##### [推多米诺](https://leetcode.cn/problems/push-dominoes/)

  ​`5.2 看题解写的`​

  ```cpp
  /*
   * @lc app=leetcode.cn id=838 lang=cpp
   *
   * [838] 推多米诺
   */

  // @lc code=start
  // 2025.5.2 每日一题
  class Solution
  {
  public:
      const int inf = 0x3f3f3f3f;
      string pushDominoes(string dominoes)
      {
          int n = dominoes.size();
          vector< int > l_min(n, inf), r_min(n, inf);
          for (int i = 0; i < n; i++)
          {
              if (dominoes[i] == '.')
              {
                  if (i != 0)
                      r_min[i] = min(r_min[i], r_min[i - 1] + 1);
              } else if (dominoes[i] == 'R')
                  r_min[i] = 0;
          }
          for (int i = n - 1; i >= 0; i--)
          {
              if (dominoes[i] == '.')
              {
                  if (i != n - 1)
                      l_min[i] = min(l_min[i], l_min[i + 1] + 1);
              } else if (dominoes[i] == 'L')
                  l_min[i] = 0;
          }
          for (int i = 0; i < n; i++)
          {
              if (l_min[i] > r_min[i])
                  dominoes[i] = 'R';
              else if (l_min[i] < r_min[i])
                  dominoes[i] = 'L';
              else
                  dominoes[i] = '.';
          }
          return dominoes;
      }
  };
  // @lc code=end
  ```

### Hard
