

def longestCommonSubstring(str1:str, str2:str) -> int :
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

def longestCommonSubsequence(A:list, B:list) -> int:
    """
    Return the length of the longest common subsequence
    of A and B.
    """
    dp = [[0 for _ in range(len(B)+1)] for _ in range(len(A)+1)]
    for i in range(1, len(A)+1):
        for j in range(1, len(B)+1):
            if A[i-1] == B[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]

if __name__ == "__main__":
    s1 = "abc"
    s2 = "abababcdd"
    print(longestCommonSubstring(s1, s2))
    A = [1,3,4,5,6,7,7,8]
    B = [3,5,7,4,8,6,7,8,2]
    print(longestCommonSubsequence(A, B))


