import csv
import sys

def main():

	# Imports the file to have the label flipped
	infile = open(sys.argv[1])

	data = []

	for row in infile:
		data.append(row)
	infile.close()

	# Splits the string into a list of lists.  Strips end lines
	for item in range(0,len(data)):
		data[item] = data[item].split(',')
		for n in range(0, len(data[item])):
			data[item][n] = data[item][n].rstrip('\n')
		x = data[item].pop(0)
		data[item].append(x)

	# Overwrites the file
	outfile = open( sys.argv[1], 'w')
	writer = csv.writer(outfile)
	writer.writerows(data)
	outfile.close()

main()