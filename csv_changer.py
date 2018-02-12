import csv,argparse

try:
	
	prompt = 'Enter input location:'
	input_name = raw_input(prompt)
	prompt = 'Enter output location:'
	output_name = raw_input(prompt)
	row_to_remove = 9
	row_to_insert = 35
	rows_to_move = [107,125]
	rows_save = []

	f_r = open(input_name)
	f_wr = open(output_name, 'wb')
	csv_r = csv.reader(f_r)		
	csv_wr = csv.writer(f_wr)



	for i,row in enumerate(csv_r):
		if i >= rows_to_move[0]-1 and i < rows_to_move[1]-1:
			rows_save.append(row)

	f_r.close()
	f_r = open(input_name)
	csv_r = csv.reader(f_r)

	for i, row in enumerate(csv_r):
	    if (i != row_to_remove-1) and not (i >= rows_to_move[0]-1 and i < rows_to_move[1]-1):
	    	if (i != row_to_insert - 1):
	        	csv_wr.writerow(row)
	    	else:
	    		for subrow in rows_save:
	    			csv_wr.writerow(subrow)
    			csv_wr.writerow(row)
		
			
			         

	f_r.close()
	f_wr.close()
	

except Exception as ex:
    print ex
    raw_input()