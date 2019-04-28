
def lcs(str1:str, str2:str) -> int :
    """ 
    Using dynamic programming to solve the longest 
    common substring problem. 
    """
    n1, n2 = len(str1), len(str2)
    dp = [[0 for _ in range(n2)] for _ in range(n1)]
    for i in range(n2):
        dp[0][i] = 1 if str1[0] == str2[i] else 0
    for i in range(n1):
        dp[i][0] = 1 if str1[i] == str2[0] else 0
    for i in range(1, n1):
        for j in range(1, n2):
            dp[i][j] = dp[i-1][j-1] + 1 if str1[i] == str2[j] else 0
    
    return max(max(_) for _ in dp)

if __name__ == "__main__":
    s1 = "abc"
    s2 = "abababcdd"
    print(lcs(s1, s2))