# small script to perform the Collatz Conjecture and graph the output
# done two ways
# with a while loop and recursive function
# mathematics researchers have run 15 quintillion numbers and yet to find an exception
# no proof yet exists for the conjecture


def collatz_whileloop(start_num):
	"""
	perform the Collatz Conjecture algorithm using a while loop
	:param start_num: input number to start with
	:return: series generated by Collatx algorithm
	"""

	collatz_series = list()
	curr_num = start_num

	collatz_series.append(curr_num)
	while curr_num != 1:
		if curr_num % 2 == 0:
			next_num = int(curr_num / 2)
		else:
			next_num = int((curr_num * 3) + 1)
		collatz_series.append(next_num)
		curr_num = next_num
	else:
		print("Collatz Sequence converged to 1 after %s steps" % len(collatz_series))

	return collatz_series


if __name__ == "__main__":
	start_num_list = list(range(2, 1000))
	for num in start_num_list:
		series = collatz_whileloop(num)
		print(series)