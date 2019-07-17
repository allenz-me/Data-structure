
public class DisjointSets {
	public int[] parent;
	private int[] size;

	public DisjointSets(int N) {
		parent = new int[N];
		size = new int[N];
		for (int i = 0; i < N; i++) {
			parent[i] = i;
			size[i] = i;
		}
	}

	private int find(int p) {
		if (p == parent[p]) {
			return p;
		} else {
			parent[p] = find(parent[p]);
			return parent[p];
		}
	}

	public isConnected(int p, int q) {
		return find(p) == find(q);
	}

	public void connect(int p, int q) {
		int pid = find(p);
		int qid = find(q);
		if (pid == qid) return;
		if (size[pid] < size[qid]) {
			parent[pid] = qid;
			size[qid] += size[pid];
		} else {
			parent[qid] = pid;
			size[pid] += size[qid];
		}
	}
}