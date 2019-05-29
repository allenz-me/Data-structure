

def nextExceed(nums:list) -> list:
	"""
	���ص�����ĵ�i��λ�õ�ֵӦ���ǣ�����ԭ�����еĵ�i��Ԫ�أ�
	���������߶��ٲ�����������һ�����Լ����Ԫ��.
	�����֮��û�б��Լ����Ԫ�أ������Ѿ������һ��Ԫ�أ�
	���ڷ�������Ķ�Ӧλ�÷���-1����
	"""
	res = [-1 for i in range(len(nums))]
	stack = []
	for i in range(len(nums)):
		while stack and nums[stack[-1]] < nums[i]:
			res[stack[-1]] = i - stack[-1]
			stack.pop()
		stack.append(i)
	return res

print(nextExceed([5,3,1,2,4]))