
/* Solve the longest common substring problem using dynamic programming. */
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

    public static void main(String[] args) {
        String s1 = "abc", s2 = "abcbceaaaaaad";
        System.out.println(Lcs.longestCommonSubstring(s1, s2));
    }
}