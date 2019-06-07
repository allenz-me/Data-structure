

class QuickUnion:

	def __init__(self, n:int):
		self.par = [i for i in range(n)]

	def find(self, p:int):
		while self.par[p] != p:
			p = self.par[p]
		return p

	def connect(self, p:int, q:int):
		pid = self.find(p)
		qid = self.find(q)
		self.par[pid] = qid

	def isConnected(self, p:int, q:int) -> bool:
		return self.find(p) == self.find(q)

qu = QuickUnion(8)
qu.connect(7, 3)
qu.connect(4, 3)
qu.connect(1, 6)
print(qu.isConnected(1, 3))
print(qu.isConnected(4, 7))