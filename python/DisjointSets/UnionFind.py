
class UnionFind:

	def __init__(self, n):
		self.par = [-1 for _ in range(n)]

	def sizeOf(self, v:int) -> int:
		return -self.par[self.find(v)]

	def find(self, v:int) -> int:
		while self.par[v] > 0:
			v = self.par[v]
		return v

	def isConnected(self, p:int, q:int) -> bool:
		return self.find(p) == self.find(q)
		
	def connect(self, p:int, q:int):
		psize, qsize = self.sizeOf(p), self.sizeOf(q)
		pid, qid = self.find(p), self.find(q)
		if psize <= qsize:
			self.par[pid] = qid
			self.par[qid] -= psize
		else:
			self.par[qid] = pid
			self.par[pid] -= qsize


uf = UnionFind(8)
uf.connect(1, 7)
uf.connect(7, 3)
print(uf.isConnected(1, 3))
print(uf.isConnected(1, 4))