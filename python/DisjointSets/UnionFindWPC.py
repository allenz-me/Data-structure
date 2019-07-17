

class UnionFindWPC:
	"""Union find with path compression"""

	def __init__(self, n):
		self.par = [-1] * n

	def sizeOf(self, v:int) -> int:
		return - self.par[self.find(v)]

	def find(self, v:int) -> int:
		if self.par[v] < 0:
			return v
		else:
			self.par[v] = self.find(self.par[v])
			return self.par[v]

	def isConnected(self, p:int, q:int) -> bool:
		return self.find(p) == self.find(q)

	def connect(self, p:int, q:int) -> None:
		pid, qid = self.find(p), self.find(q)
		psize, qsize = - self.par[pid], - self.par[qid]
		if pid == qid:
			return
		if psize <= qsize:
			self.par[pid] = qid
			self.par[qid] -= self.par[pid]
		else:
			self.par[qid] = pid
			self.par[pid] -= self.par[qid]

uf = UnionFindWPC(4)
uf.connect(2, 0)
uf.connect(0, 2)
uf.connect(3, 0)
uf.connect(2, 1)
print(uf.isConnected(1, 3))
print(uf.isConnected(1, 2))