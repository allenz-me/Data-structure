
import abc

class DisjointSets(metaclass=abc.ABCMeta):

	@abc.abstractmethod
	def connect(self, p:int, q:int):
		"""connect two points"""
		pass

	@abc.abstractmethod
	def isConnected(self, p:int, q:int) -> bool:
		"""judge if p and q are connected"""
		pass
