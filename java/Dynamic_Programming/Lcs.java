
/* Solve the longest common substring, subsequence problem using dynamic programming. */
public class Lcs {
    /* Return the length of the longest common substring of string str1, str2 */
    public static int longestCommonSubstring(String str1, String str2) {
        int n1 = str1.length(), n2 = str2.length();
        if (n1 == 0 || n2 == 0) {
            return 0;
        }
        int[][] dp = new int[n1][n2];
        int res = 0;
        // 二维表的边界值, 也可以让 dp = int[n1+1][n2+1], 将边界值设为0, 
        for (int i=0; i<n2; i++) {
            dp[0][i] = (str1.charAt(0) == str2.charAt(i)) ? 1 : 0;
            res = Math.max(res, dp[0][i]);
        }
        for (int i=0; i<n1; i++) {
            dp[i][0] = (str1.charAt(i) == str2.charAt(0)) ? 1 : 0;
            res = Math.max(res, dp[i][0]);
        }

        for (int i=1; i<n1; i++) {
            for (int j=1; j<n2; j++) {
                dp[i][j] = (str1.charAt(i) == str2.charAt(j)) ? dp[i-1][j-1] + 1 : 0;
                res = Math.max(dp[i][j], res);
            }
        }
        return res;
    }

    /* Return the length of the longest common subsequence of A and B. */
    public static int longestCommonSubsequence(int[] A, int[] B) {
        int la = A.length, lb = B.length;
        int[][] dp = new int[la+1][lb+1];
        for (int i=1; i<la+1; i++) {
            for (int j=1; j<lb+1; j++) {
                if (A[i-1] == B[j-1]) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }
        return dp[la][lb];
    }

    public static void main(String[] args) {
        String s1 = "abcd", s2 = "abcbceaaaaaad";
        System.out.println(Lcs.longestCommonSubstring(s1, s2));
        int[] A = {1,3,4,5,6,7,7,8};
        int[] B = {3,5,7,4,8,6,7,8,2};
        System.out.println(Lcs.longestCommonSubsequence(A, B));
    }
}