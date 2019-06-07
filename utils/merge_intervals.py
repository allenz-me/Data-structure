

def mergeIntervals(intervals):
	"""合并区间 intervals: List[List[int]]"""
	if len(intervals) <= 1:
		return intervals
	res = []
	# 按区间的第一位排序,当列表已经排好序，能够合并的区间构成了连通块。
	intervals.sort(key=lambda x: x[0])
	start, end = intervals[0] # 这里是序列解包
	for i, l in enumerate(intervals, start=1):
		if end < l[0]:
			res.append([start, end])
			start, end = l
		else:
			end = max(end, l[1])
	if res and res[-1] == [start, end]:
		return res
	res.append([start, end])
	return res