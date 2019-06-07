

def lev(word1:str, word2:str) -> int:
    """计算两个字符串的编辑距离: Levenshtein Distance"""
    n1, n2 = len(word1), len(word2)
    if n1 == 0 or n2 == 0:
        return n1 if n1 else n2
    dp = [[0 for _ in range(n2+1)] for _ in range(n1+1)]
    for i in range(1, n1+1):
        dp[i][0] = i
    for j in range(1, n2+1):
        dp[0][j] = j
    for i in range(1, n1+1):
        for j in range(1, n2+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]-1)
                # 实际上就是: dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n1][n2]

word1 = "horse"
word2 = "housing"

print(lev(word1, word2)) # 4