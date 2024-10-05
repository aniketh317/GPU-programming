from os.path import exists
import sys
import random
import csv
import time

def main(input_id, num_elements, num_iterations=1):
  input_lock_file_name = "input_" + input_id + ".lock"
  input_file_name = "input_" + input_id + ".csv"
  output_lock_file_name = "output_" + 'c' + ".lock"
  output_file_name = "output_" + input_id + ".csv"
  # Update this and possibly the CUDA code to take as an argument for how many runs will be executed
  for i in range(num_iterations):
    output_csv_line(input_file_name=input_file_name, num_elements=num_elements)
    input_lock_file = open(input_lock_file_name, 'w')
    input_lock_file.write(str(i))
    input_lock_file.close()
    while not exists(output_lock_file_name):
      time.sleep(5.0)
    output_iteration = -1
    while(output_iteration!=i):
      output_lock_file = open(output_lock_file_name, 'r')
      reader = csv.reader(output_lock_file, delimiter=',', quotechar='|')
      output_iteration = int(next(reader)[0])
      output_lock_file.close()
      time.sleep(5.0)
    read_output_csv_file(output_file_name=output_file_name, num_elements=num_elements)
    time.sleep(5.0)


def output_csv_line(input_file_name, num_elements):
  f = open(input_file_name,'w')
  input_data = []
  for _ in range(num_elements):
    input_data.append(random.randint(0, num_elements))
  # Below is based on https://www.geeksforgeeks.org/python-list-of-float-to-string-conversion/
  f.write(','.join([str(i) for i in input_data]))
  f.close()


def read_output_csv_file(output_file_name, num_elements):
  with open(output_file_name, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    output_data = next(reader)
    print(output_data)


if __name__ == "__main__":
  argc = len(sys.argv)
  if argc > 2:
    input_id = sys.argv[1]
    num_elements = int(sys.argv[2])
    num_iterations = 1
    if argc > 3:
      num_iterations = int(sys.argv[3])
    main(input_id=input_id, num_elements=num_elements, num_iterations=num_iterations)
