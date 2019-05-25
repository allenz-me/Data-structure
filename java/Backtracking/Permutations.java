import java.util.Collections;
import java.util.List;
import java.util.ArrayList;

/* Given a collection of distinct integers, 
 return all possible permutations. */
public class Permutations {

	public List<List<Integer>> permute(int[] nums) {
		List<List<Integer>> res = new ArrayList<>();
		List<Integer> list = new ArrayList<>();
		for (int num : nums) {
			list.add(num);
		}
		helper(res, list, 0, nums.length);
		return res;
	}

	private void helper(List<List<Integer>> res, List<Integer> temp, int first, int n) {
		if (first == n) {
			res.add(new ArrayList<Integer>(temp));
		}
		for (int i = first; i < n; i++) {
			Collections.swap(temp, first, i);
			helper(res, temp, first + 1, n);
			Collections.swap(temp, first, i);
		}
	}

	public static void main(String[] args) {
		int[] nums = {1, 4, 7};
		Permutations p = new Permutations();
		List<List<Integer>> res = p.permute(nums);
		System.out.println(res);
	}
}