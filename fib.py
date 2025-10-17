
n = 9
previous_sum = 1
previous_sum_2 = 2
for i in range(2, n):
    new_sum = previous_sum + previous_sum_2
    previous_sum = previous_sum_2
    previous_sum_2 = new_sum

print (new_sum)
