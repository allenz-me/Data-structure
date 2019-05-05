

# 纯python递归计算行列式

def det(array:list) -> int:
	"""
	type array : List[List[float]]
	"""
	assert len(array) == len(array[0])
	if len(array) == 1:
		return array[0][0]
	# 沿第一列展开
	s = 0
	for i in range(len(array)):
		# 余子式
		A = [array[j][1:] for j in range(len(array)) if j != i]
		print(A)
		if i % 2:
			s -= array[i][0] * det(A)
		else:
			s += array[i][0] * det(A)
	return s

l = [[2,2,2],[2,2,9],[3,6,3]]
print(det(l))