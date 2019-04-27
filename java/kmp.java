
public class KMP {
	
	/* Return the prefix table of string s */
	public static int[] prefix_table(String s) {
		int[] next = new int[s.length()];
		next[0] = -1;
		int j = 0, k = -1;
		while (j < s.length() - 1) {
			if (k == -1 || s.charAt(j) == s.charAt(k)) {
				next[++j] = ++k;
			} else {
				k = next[k];
			}
		}
		return next;
	}
	
	/* Return the first index if the pattern matches the target else -1 */
	public static int kmp(String target, String pattern) {
		int[] next = prefix_table(pattern);
		int t = 0, p = 0; // points to the target and the pattern repectively.
		while (t < target.length() && p < pattern.length()) {
			if (p == -1 || target.charAt(t) == pattern.charAt(p)) {
				p++;
				t++;
			} else {
				p = next[p];
			}
		}
		if (p == pattern.length()) {
			return t - p;
		} else {
			return -1;
		}
	}
	
}
			
		
}