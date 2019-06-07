
class QuickFind:

	def __init__(self, n):
		self.par = [i for i in range(n)]

	def connect(self, p:int, q:int):
		pid = self.par[p]
		qid = self.par[q]
		for i in range(len(self.par)):
			if self.par[i] == pid:
				self.par[i] = qid

	def isConnected(self, p:int, q:int) -> bool:
		return self.par[p] == self.par[q]


qf = QuickFind(10)
qf.connect(1, 2)
qf.connect(4, 5)
qf.connect(2, 4)
print(qf.isConnected(1, 5))
print(qf.isConnected(1, 9))