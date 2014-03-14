import sys
import random
import numpy as np

data = []

def split(data):
	max_jump = 0
	jump_index = -1
	for i in range(1, len(data)):
		jump = data[i] - data[i-1]
		if jump >= max_jump:
			max_jump = jump
			jump_index = i
	
	data_left = data[0:jump_index]
	data_right = data[jump_index:]
	return data_left, data_right

if __name__ == "__main__":
	for i in range(1000):
		data.append(random.randint(1, 200))

	data.sort()
	buckets = []
	
	data_left = data
	while len(data_left) >= 2:
		data_left, data_right = split(data_left)
		buckets.append(data_right)

	print buckets
